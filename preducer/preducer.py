#!/usr/bin/env python3

"""
Python script to downgrade a double precision Fortran routine to single
(real4) precision without changing its external signature, for example
to study the effect of reduced precision arithmetic within only one
subroutine that is part of a larger piece of software.

The script takes the following arguments:
    1. The file name to be read. A will be created with _real4 appended
       to the name.
    2. An optional argument specifying the name of the subroutine that is
       to be treated. If the argument is not given, all subroutines in
       the file will be modified.

Limitations:
 - Currently only F77 files are supported. This could be easily fixed by
   using the fortran.two parser for F90 files and adjusting some of the
   node types, e.g. not only test for fparser.one.statements.Assignment
   but also for f2003.Assignment_Stmt
 - Currently only files with one subroutine and nothing else are
   supported. No modules, no classes, nothing. This could also easily
   fixed, by discovering subroutine nodes in a recursive AST search just
   like what is already happening to discover assignments
 - Whether a variable is first written or first read is determined
   lexicographically, not by building the flow graph. This means that
   branches or even goto statements can trick this analysis. It should
   still be enough for cases where variables are either read-only or
   write-only. Fixing this would be more difficult.
 - Read and write access to variables is only detected in assignments,
   for example in 'foo(i) = bar + z(g)' we would detect 'foo' as being
   written, and 'bar', 'z' as being read. No other access is detected,
   for example a subroutine or function call would not result in the
   arguments being added to the read and write lists.
"""

"""
try:
    from pip._internal.operations import freeze
except ImportError:  # pip < 10.0
    from pip.operations import freeze

freezer = freeze.freeze()
for p in freezer:
    print(p)
"""

import sys, re
import textwrap
import subprocess
import fparser.one.parsefortran
import fparser.common.readfortran
from shutil import copyfile


verbose = False
unitname = None

def printv(arg):
    if verbose:
        print(arg)


def cleanVariableName(var):
    """
    A reference to a variable can be something messy like "BaR(3,foo),", where
    all we want is the variable name "bar". This function removes array
    indices, trailing commas etc, and makes everything lowercase, to get only
    a clean variable name and nothing else.
    """
    return re.split('[,(]',var)[0].lower()

def find_vars(varstring):
    current_varname = ''
    varlist = list()
    parentheses_depth = 0
    for i, c in enumerate(varstring):
        if c == '(':
            parentheses_depth += 1
        elif c == ')':
            parentheses_depth -= 1
        elif parentheses_depth == 0:
            if c != ' ' and c != '\t':
                if(c == ','):
                    if(len(current_varname.strip())>0):
                        varlist.append(current_varname)
                        current_varname = ''
                else:
                    current_varname += c
    varlist.append(current_varname)
    if(varlist[0].lower()=='doubleprecision'):
        del(varlist[0])
    else:
        varlist[0] = varlist[0][15:]
    return varlist

def visitDoublePrecisionStmt(node):
    """
    The f77 parser treats a line containing a double precision variable
    declaration as a Line, which is a string of characters. We need to extract
    the variable names from that string, and not get confused by arrays. For
    example, "double precision foo(a,3), bar" should give us the variables
    "foo" and "bar", and nothing else.
    """
    if(type(node)!=fparser.one.typedecl_statements.DoublePrecision):
        raise Exception("visitDoublePrecisionStmt called on wrong node type")
    slist = find_vars(node.item.line)
    varset = set()
    for s in slist:
        varname = cleanVariableName(s)
        varset.add(varname) # add this variable name to set
    return varset

def visitNode(node,doublevars,doublevars_modified):
    """
    Recursively go through the AST and find all assignments.
    This is needed to find variables that are read before modified, and
    variables that are modified at all.
    """
    children = []
    doublevars_predefined = set()
    if hasattr(node, "content"):
        children = node.content
    elif hasattr(node, "items"):
        children = node.items
    elif type(node) in (tuple, list):
        children = node
    for child in children:
        if(type(child)==fparser.one.statements.Assignment):
            lhs = cleanVariableName(child.variable)
            # Visit an assignment statement, e.g. "a = b + c"
            if(lhs in doublevars):
                doublevars_modified.add(lhs)
            rhs = child.expr
            readDoubleVars = set(filter(lambda x: x in rhs, doublevars))
            doublevars_predefined = doublevars_predefined.union(readDoubleVars.difference(doublevars_modified))
        else:
            newmodified, newpredefined = visitNode(child, doublevars, doublevars_modified)
            doublevars_modified = doublevars_modified.union(newmodified)
            doublevars_predefined = doublevars_predefined.union(newpredefined)
    return doublevars_modified, doublevars_predefined

def f77linebreaks(instr):
    """
    Takes a string as an input, and breaks all lines after at most 72
    characters, using F77 line continuation markers.
    """
    outstr = ''
    for l in instr.splitlines():   #loop through all lines
        if(len(l.strip())==0): # empty line
            outstr += l+'\n'   # just add pages break
        elif(l[0]!=' ' or l.lstrip()[0]=='!'): # comment line, never touch those
            outstr += l+'\n'
        else:
            if(len(l) > 7 and l[0:7].strip().isnumeric()): # workaround for parser bug: numeric line labels are printed with an incorrect blank space in column 1. Remove this.
                l = l[0:7].strip().ljust(7) + l[7:]
            while(len(l) > 72):
                outstr += l[0:71]+'\n'
                l = '     *'+l[71:]
            outstr += l+'\n'
    return outstr

def real4subroutine(unit, file, allunits):
    # Analysis part: Find the subroutine that needs to be modified,
    # and for that subroutine, find the double precision arguments
    # and for each of those, find out whether they are in/outputs.
    args = unit.args.copy()
    if(unit.blocktype == 'function'):
        args.append(unit.name)
    printv(args)
    doublevars = set()            # all double precision variables declared within subroutine
    doublevars_predefined = set() # all double precision variables read before being modified
    doublevars_modified = set()   # all double precision variables modified within subroutine
    decls = list()
    for c in unit.content:
        decltypes = [fparser.one.typedecl_statements.Byte,
                     fparser.one.typedecl_statements.Character,
                     fparser.one.typedecl_statements.Complex,
                     fparser.one.typedecl_statements.DoubleComplex,
                     fparser.one.typedecl_statements.DoublePrecision,
                     fparser.one.typedecl_statements.Integer,
                     fparser.one.typedecl_statements.Logical,
                     fparser.one.typedecl_statements.Real,
                     fparser.one.statements.Parameter]
        if(type(c) in decltypes):
            decls.append(c)
            if(type(c) == fparser.one.typedecl_statements.DoublePrecision):
                doublevars = doublevars.union(visitDoublePrecisionStmt(c))
        else:
            newmodified, newpredefined = visitNode(c, doublevars, doublevars_modified)
            doublevars_modified = doublevars_modified.union(newmodified)
            doublevars_predefined = doublevars_predefined.union(newpredefined)
    doubleargs_modified = doublevars_modified.intersection(args)
    doubleargs_predefined = doublevars_predefined.intersection(args)
    printv("local double precision variables: %s"%doublevars.difference(args).__str__())
    printv("double precision arguments: %s"%doublevars.intersection(args).__str__())
    printv(" - modified: %s"%(doubleargs_modified.__str__()))
    printv(" - input: %s"%(doubleargs_predefined.__str__()))
    printv(" - unused: %s"%(doublevars.intersection(args).difference(doubleargs_predefined.union(doubleargs_modified)).__str__()))

    # Cloning part: Create a subroutine that has the same body as the original
    # one, but uses the new precision throughout and append _sp to its name
    fclone = unit.tofortran()
    fclone = fclone.replace('DOUBLEPRECISION','REAL')
    if(unit.blocktype == 'function'):
        fclone = re.sub('FUNCTION %s'%unit.name,'FUNCTION %s_sp'%unit.name, fclone, flags=re.IGNORECASE)
    else:
        fclone = re.sub('SUBROUTINE %s'%unit.name,'SUBROUTINE %s_sp'%unit.name, fclone, flags=re.IGNORECASE)
    for otherunit in allunits:
        fclone = re.sub('CALL %s\('%otherunit.name, 'CALL %s_sp('%otherunit.name, fclone, flags=re.IGNORECASE)
    fclone = re.sub('1.0d308', '1.0e38', fclone, flags=re.IGNORECASE)
    fclone = f77linebreaks(fclone)
    file.write(fclone)
    file.write('\n\n')

    # Wrapper part: Create a subroutine that has the signature of the original
    # one, and performs the down-cast/call/up-cast to the reduced precision
    # subroutine.
    args_str = ", ".join(unit.args)
    args_sp = args_str
    for dv in doublevars:
        args_sp = re.sub(r"\b%s\b" % dv , '%s_sp'%dv, args_sp)
    decls_sp = list()
    for d in decls:
        if(type(d) == fparser.one.typedecl_statements.DoublePrecision):
            varnames = visitDoublePrecisionStmt(d)
            d_sp = d.item.line.replace('DOUBLE PRECISION','REAL').lower()
            for vn in varnames:
                d_sp = re.sub(r"\b%s\b" % vn , '%s_sp'%vn, d_sp)
            decls_sp.append(d_sp)
        decls_sp.append(d.item.line)
    decls_sp = "\n".join(decls_sp)
    copyin = set()
    for dm in doubleargs_predefined:
        copyin.add("%s_sp = %s"%(dm,dm))
    copyin = "\n".join(copyin)
    copyout = set()
    for dm in doubleargs_modified:
        copyout.add("%s = %s_sp"%(dm,dm))
    copyout = "\n".join(copyout)
    if(unit.blocktype == 'function'):
        wrapper = "double precision function %s(%s)\n%s\n%s\n%s = %s_sp(%s)\n%s\nreturn\nend function"%(unit.name,args_str,decls_sp,copyin,unit.name,unit.name,args_sp,copyout)
    else:
        wrapper = "subroutine %s(%s)\n%s\n%s\ncall %s_sp(%s)\n%s\nend subroutine"%(unit.name,args_str,decls_sp,copyin,unit.name,args_sp,copyout)
    wrapper = f77linebreaks(textwrap.indent(wrapper,7*' '))
    file.write(wrapper)

def run_preducer(filename):
    filename_preduced = "%s_preduced.f"%(filename[0:-2])

    # Parse Fortran file
    reader = fparser.common.readfortran.FortranFileReader(filename)
    fp = fparser.one.parsefortran.FortranParser(reader)
    fp.parse()

    if(len(fp.block.content) == 0):
        if filename != "EXTER.f":
            print("Warning: Preducer called on empty file %s"%(filename))
        copyfile(filename, filename_preduced)
        #exit()
        return None
    with open(filename_preduced,'w') as file:
        for unit in fp.block.content:
            if(unit.blocktype != 'subroutine' and unit.blocktype != 'function'):
                raise Exception("Top Unit is neither subroutine nor function")
            if(unitname == None or unit.name == unitname):
                real4subroutine(unit, file, fp.block.content)
