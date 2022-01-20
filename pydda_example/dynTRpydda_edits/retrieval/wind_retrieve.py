"""
Created on Mon Aug  7 09:17:40 2017

@author: rjackson
"""

import pyart
import numpy as np
from numpy.linalg import norm
import time
import cartopy.crs as ccrs
import math
import sys, os


import DynPrec
import jax
import pandas as pd
import copy

#global prev_prec
prev_prec = 1

#imports changed to local import path to run on computer
import cost_functions
from cost_functions.cost_functions import J_function, grad_J
from scipy.optimize import fmin_l_bfgs_b
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from matplotlib import pyplot as plt
from copy import deepcopy
from retrieval.angles import add_azimuth_as_field, add_elevation_as_field



# Define multiprecision function
#def obj_fun(z, parameters1, precision, jax):
def obj_fun(z, params, precision, jax_on):
    """
    if prec == 1:
        jax.config.update("jax_enable_x64", False)
    elif prec == 2:
        jax.config.update("jax_enable_x64", True)
    else:
        print("Can't recognize precision", prec)
    """

    fhandle = J_function(z, params, precision, jax_on)
    ghandle = grad_J(z, params, precision, jax_on)
    return fhandle, ghandle



class DDParameters(object):
    """
    This is a helper class for inserting more arguments into the :func:`pydda.cost_functions.J_function` and
    :func:`pydda.cost_functions.grad_J` function. Since these cost functions take numerous parameters, this class
    will store the needed parameters as one positional argument for easier readability of the code.

    In addition, class members can be added here so that those contributing more constraints to the variational
    framework can add any parameters they may need.

    Attributes
    ----------
    vrs: List of float arrays
        List of radial velocities from each radar
    azs: List of float arrays
        List of azimuths from each radar
    els: List of float arrays
        List of elevations from each radar
    wts: List of float arrays
        Float array containing fall speed from radar.
    u_back: 1D float array (number of vertical levels)
        Background u wind
    v_back: 1D float array (number of vertical levels)
        Background u wind
    u_model: list of 3D float arrays
        U from each model integrated into the retrieval
    v_model: list of 3D float arrays
        V from each model integrated into the retrieval
    w_model:
        W from each model integrated into the retrieval
    Co: float
        Weighting coefficient for data constraint.
    Cm: float
        Weighting coefficient for mass continuity constraint.
    Cx: float
        Smoothing coefficient for x-direction
    Cy: float
        Smoothing coefficient for y-direction
    Cz: float
        Smoothing coefficient for z-direction
    Cb: float
        Coefficient for sounding constraint
    Cv: float
        Weight for cost function related to vertical vorticity equation.
    Cmod: float
        Coefficient for model constraint
    Cpoint: float
        Coefficient for point constraint
    Ut: float
        Prescribed storm motion. This is only needed if Cv is not zero.
    Vt: float
        Prescribed storm motion. This is only needed if Cv is not zero.
    grid_shape:
        Shape of wind grid
    dx:
        Spacing of grid in x direction
    dy:
        Spacing of grid in y direction
    dz:
        Spacing of grid in z direction
    x:
        E-W grid levels in m
    y:
        N-S grid levels in m
    z:
        Grid vertical levels in m
    rmsVr: float
        The sum of squares of velocity/num_points. Use for normalization
        of data weighting coefficient
    weights: n_radars by z_bins by y_bins x x_bins float array
        Data weights for each pair of radars
    bg_weights: z_bins by y_bins x x_bins float array
        Data weights for sounding constraint
    model_weights: n_models by z_bins by y_bins by x_bins float array
        Data weights for each model.
    point_list: list or None
        point_list: list of dicts
        List of point constraints. Each member is a dict with keys of "u", "v",
        to correspond to each component of the wind field and "x", "y", "z"
        to correspond to the location of the point observation in the Grid's
        Cartesian coordinates.
    roi: float
        The radius of influence of each point observation in m.
    upper_bc: bool
        True to enforce w=0 at top of domain (impermeability condition),
        False to not enforce impermeability at top of domain
    """
    def __init__(self):
        self.Ut = np.nan
        self.Vt = np.nan
        self.rmsVr = np.nan
        self.grid_shape = None
        self.Cmod = np.nan
        self.Cpoint = np.nan
        self.u_back = None
        self.v_back = None
        self.wts = []
        self.vrs = []
        self.azs = []
        self.els = []
        self.weights = []
        self.bg_weights = []
        self.model_weights = []
        self.u_model = []
        self.v_model = []
        self.w_model = []
        self.x = None
        self.y = None
        self.z = None
        self.dx = np.nan
        self.dy = np.nan
        self.dz = np.nan
        self.Co = 1.0
        self.Cm = 1500.0
        self.Cx = 0.0
        self.Cy = 0.0
        self.Cz = 0.0
        self.Cb = 0.0
        self.Cv = 0.0
        self.Cmod = 0.0
        self.Cpoint = 0.0
        self.Ut = 0.0
        self.Vt = 0.0
        self.upper_bc = True
        self.roi = 1000.0
        self.frz = 4500.0
        self.point_list = []


def get_dd_wind_field(Grids, u_init, v_init, w_init, points=None, vel_name=None,
                      refl_field=None, u_back=None, v_back=None, z_back=None,
                      frz=4500.0, Co=1.0, Cm=1500.0, Cx=0.0,
                      Cy=0.0, Cz=0.0, Cb=0.0, Cv=0.0, Cmod=0.0, Cpoint=0.0,
                      Ut=None, Vt=None, filt_iterations=2,
                      mask_outside_opt=False, weights_obs=None,
                      weights_model=None, weights_bg=None,
                      max_iterations=10000, mask_w_outside_opt=True,
                      filter_window=9, filter_order=3, min_bca=30.0,
                      max_bca=150.0, upper_bc=True, model_fields=None,
                      output_cost_functions=True, roi=1000.0,
                      max_memory=10, use_dynTR=True, store_history=False, gtol=1e-6,
                      precision_dict=None, subproblem_tol=1e-6,
                      write_folder=None, jax_on=True):
    """
    This function takes in a list of Py-ART Grid objects and derives a
    wind field. Every Py-ART Grid in Grids must have the same grid
    specification. 

    In order for the model data constraint to be used,
    the model data must be added as a field to at least one of the
    grids in Grids. This involves interpolating the model data to the
    Grids' coordinates. There are helper functions for this for WRF
    and HRRR data in :py:func:`pydda.constraints`:

    :py:func:`make_constraint_from_wrf`

    :py:func:`add_hrrr_constraint_to_grid`

    Parameters
    ==========

    Grids: list of Py-ART Grids
        The list of Py-ART grids to take in corresponding to each radar.
        All grids must have the same shape, x coordinates, y coordinates
        and z coordinates.
    u_init: 3D ndarray
        The intial guess for the zonal wind field, input as a 3D array
        with the same shape as the fields in Grids.
    v_init: 3D ndarray
        The intial guess for the meridional wind field, input as a 3D array
        with the same shape as the fields in Grids.
    w_init: 3D ndarray
        The intial guess for the vertical wind field, input as a 3D array
        with the same shape as the fields in Grids.
    points: None or list of dicts
        Point observations as returned by :func:`pydda.constraints.get_iem_obs`. Set
        to None to disable.
    vel_name: string
        Name of radial velocity field. Setting to None will have PyDDA attempt
        to automatically detect the velocity field name.
    refl_field: string
        Name of reflectivity field. Setting to None will have PyDDA attempt 
        to automatically detect the reflectivity field name.
    u_back: 1D array
        Background zonal wind field from a sounding as a function of height.
        This should be given in the sounding's vertical coordinates.
    v_back: 1D array
        Background meridional wind field from a sounding as a function of
        height. This should be given in the sounding's vertical coordinates.
    z_back: 1D array
        Heights corresponding to background wind field levels in meters. This
        is given in the sounding's original coordinates.
    frz: float
        Freezing level used for fall speed calculation in meters.
    Co: float
        Weight for cost function related to observed radial velocities.
    Cm: float
        Weight for cost function related to the mass continuity equation.
    Cx: float
        Weight for cost function related to smoothness in x direction
    Cy: float
        Weight for cost function related to smoothness in y direction
    Cz: float
        Weight for cost function related to smoothness in z direction
    Cv: float
        Weight for cost function related to vertical vorticity equation.
    Cmod: float
        Weight for cost function related to custom constraints.
    Cpoint: float
        Weight for cost function related to point observations.
    weights_obs: list of floating point arrays or None
        List of weights for each point in grid from each radar in Grids.
        Set to None to let PyDDA determine this automatically.
    weights_model: list of floating point arrays or None
        List of weights for each point in grid from each custom field in
        model_fields. Set to None to let PyDDA determine this automatically.
    weights_bg: list of floating point arrays or None
        List of weights for each point in grid from the sounding. Set to None
        to let PyDDA determine this automatically.
    Ut: float
        Prescribed storm motion in zonal direction.
        This is only needed if Cv is not zero.
    Vt: float
        Prescribed storm motion in meridional direction.
        This is only needed if Cv is not zero.
    filt_iterations: int
        If this is greater than 0, PyDDA will run a low pass filter on
        the retrieved wind field and then do the optimization step for
        filt_iterations iterations. Set to 0 to disable the low pass filter.
    mask_outside_opt: bool
        If set to true, wind values outside the multiple doppler lobes will
        be masked, i.e. if less than 2 radars provide coverage for a given
        point.
    max_iterations: int
        The maximum number of iterations to run the optimization loop for.
    mask_w_outside_opt: bool
        If set to true, vertical winds outside the multiple doppler lobes will
        be masked, i.e. if less than 2 radars provide coverage for a given
        point.
    filter_window: int
        Window size to use for the low pass filter. A larger window will
        increase the number of points factored into the polynomial fit for
        the filter, and hence will increase the smoothness.
    filter_order: int
        The order of the polynomial to use for the low pass filter. Higher
        order polynomials allow for the retention of smaller scale features
        but may also not remove enough noise.
    min_bca: float
        Minimum beam crossing angle in degrees between two radars. 30.0 is the
        typical value used in many publications.
    max_bca: float
        Minimum beam crossing angle in degrees between two radars. 150.0 is the
        typical value used in many publications.
    upper_bc: bool
        Set this to true to enforce w = 0 at the top of the atmosphere. This is
        commonly called the impermeability condition.
    model_fields: list of strings
        The list of fields in the first grid in Grids that contain the custom
        data interpolated to the Grid's grid specification. Helper functions
        to create such gridded fields for HRRR and NetCDF WRF data exist
        in ::pydda.constraints::. PyDDA will look for fields named U_(model
        field name), V_(model field name), and W_(model field name). For
        example, if you have U_hrrr, V_hrrr, and W_hrrr, then specify ["hrrr"]
        into model_fields.
    output_cost_functions: bool
        Set to True to output the value of each cost function every
        10 iterations.
    roi: float
        Radius of influence for the point observations. The point observation will
        not hold any weight outside this radius.

    Returns
    =======
    new_grid_list: list
        A list of Py-ART grids containing the derived wind fields. These fields
        are displayable by the visualization module.
    """
    if store_history:
        if write_folder is None:
            write_folder = os.getcwd()
        print('\n \nWriting to ' + write_folder + '\n \n')

    global i
    i = 0

    # We have to have a prescribed storm motion for vorticity constraint
    if(Ut is None or Vt is None):
        if(Cv != 0.0):
            raise ValueError(('Ut and Vt cannot be None if vertical ' +
                              'vorticity constraint is enabled!'))

    if not isinstance(Grids, list):
        raise ValueError('Grids has to be a list!')

    parameters = DDParameters()
    parameters.Ut = Ut
    parameters.Vt = Vt
    
    # Ensure that all Grids are on the same coordinate system
    prev_grid = Grids[0]
    for g in Grids:
        if not np.allclose(
            g.x['data'], prev_grid.x['data'], atol=10):
            raise ValueError('Grids do not have equal x coordinates!')

        if not np.allclose(
            g.y['data'], prev_grid.y['data'], atol=10):
            raise ValueError('Grids do not have equal y coordinates!')

        if not np.allclose(
            g.z['data'], prev_grid.z['data'], atol=10):
            raise ValueError('Grids do not have equal z coordinates!')

        if not g.origin_latitude['data'] == prev_grid.origin_latitude['data']:
            raise ValueError(("Grids have unequal origin lat/lons!"))

        prev_grid = g

    # Disable background constraint if none provided
    if(u_back is None or v_back is None):
        parameters.u_back = np.zeros(u_init.shape[0])
        parameters.v_back = np.zeros(v_init.shape[0])
    else:
        # Interpolate sounding to radar grid
        print('Interpolating sounding to radar grid')
        u_interp = interp1d(z_back, u_back, bounds_error=False)
        v_interp = interp1d(z_back, v_back, bounds_error=False)
        parameters.u_back = u_interp(Grids[0].z['data'])
        parameters.v_back = v_interp(Grids[0].z['data'])
        print('Interpolated U field:')
        print(parameters["u_back"])
        print('Interpolated V field:')
        print(parameters["v_back"])
        print('Grid levels:')
        print(Grids[0].z['data'])

    # Parse names of velocity field
    if refl_field is None:
        refl_field = pyart.config.get_field_name('reflectivity')

    # Parse names of velocity field
    if vel_name is None:
        vel_name = pyart.config.get_field_name('corrected_velocity')
    winds = np.stack([u_init, v_init, w_init])


    # Set up wind fields and weights from each radar
    parameters.weights = np.zeros(
        (len(Grids), u_init.shape[0], u_init.shape[1], u_init.shape[2]))

    parameters.bg_weights = np.zeros(v_init.shape)
    if(model_fields is not None):
        parameters.model_weights = np.ones(
            (len(model_fields), u_init.shape[0], u_init.shape[1],
             u_init.shape[2]))
    else:
        parameters.model_weights = np.zeros(
            (1, u_init.shape[0], u_init.shape[1], u_init.shape[2]))

    if(model_fields is None):
        if(Cmod != 0.0):
            raise ValueError(
                 'Cmod must be zero if model fields are not specified!')

    bca = np.zeros(
        (len(Grids), len(Grids), u_init.shape[1], u_init.shape[2]))
    sum_Vr = np.zeros(len(Grids))

    for i in range(len(Grids)):
        parameters.wts.append(cost_functions.calculate_fall_speed(Grids[i],
                                                       refl_field=refl_field, frz=frz))
        add_azimuth_as_field(Grids[i], dz_name=refl_field)
        add_elevation_as_field(Grids[i], dz_name=refl_field)
        parameters.vrs.append(Grids[i].fields[vel_name]['data'])
        parameters.azs.append(Grids[i].fields['AZ']['data']*np.pi/180)
        parameters.els.append(Grids[i].fields['EL']['data']*np.pi/180)

    if(len(Grids) > 1):
        for i in range(len(Grids)):
            for j in range(i+1, len(Grids)):
                print(("Calculating weights for radars " + str(i) +
                       " and " + str(j)))
                bca[i, j] = get_bca(Grids[i].radar_longitude['data'],
                                    Grids[i].radar_latitude['data'],
                                    Grids[j].radar_longitude['data'],
                                    Grids[j].radar_latitude['data'],
                                    Grids[i].point_x['data'][0],
                                    Grids[i].point_y['data'][0],
                                    Grids[i].get_projparams())

                for k in range(parameters.vrs[i].shape[0]):
                    if(weights_obs is None):
                        cur_array = parameters.weights[i, k]
                        cur_array[np.logical_and(
                            ~parameters.vrs[i][k].mask,
                            np.logical_and(
                                bca[i, j] >= math.radians(min_bca),
                                bca[i, j] <= math.radians(max_bca)))] += 1
                        parameters.weights[i, k] = cur_array
                    else:
                        parameters.weights[i, k] = weights_obs[i][k, :, :]

                    if(weights_obs is None):
                        cur_array = parameters.weights[j, k]
                        cur_array[np.logical_and(
                            ~parameters.vrs[j][k].mask,
                            np.logical_and(
                                bca[i, j] >= math.radians(min_bca),
                                bca[i, j] <= math.radians(max_bca)))] += 1
                        parameters.weights[j, k] = cur_array
                    else:
                        parameters.weights[j, k] = weights_obs[j][k, :, :]

                    if(weights_bg is None):
                        cur_array = parameters.bg_weights[k]
                        cur_array[np.logical_or(
                            bca[i, j] >= math.radians(min_bca),
                            bca[i, j] <= math.radians(max_bca))] = 1
                        cur_array[parameters.vrs[i][k].mask] = 0
                        parameters.bg_weights[i] = cur_array
                    else:
                        parameters.bg_weights[i] = weights_bg[i]

        print("Calculating weights for models...")
        coverage_grade = parameters.weights.sum(axis=0)
        coverage_grade = coverage_grade/coverage_grade.max()

        # Weigh in model input more when we have no coverage
        # Model only weighs 1/(# of grids + 1) when there is full
        # Coverage
        if(model_fields is not None):
            if(weights_model is None):
                for i in range(len(model_fields)):
                    parameters.model_weights[i] = 1 - (coverage_grade/(len(Grids)+1))
            else:
                for i in range(len(model_fields)):
                    parameters.model_weights[i] = weights_model[i]
    else:
        if weights_obs is None:
            parameters.weights[0] = np.where(~parameters.vrs[0].mask, 1, 0)
        else:
            parameters.weights[0] = weights_obs[0]

        if weights_bg is None:
            parameters.bg_weights = np.where(~parameters.vrs[0].mask, 0, 1)
        else:
            parameters.bg_weights = weights_bg


    parameters.weights[parameters.weights > 0] = 1
    parameters.bg_weights[parameters.bg_weights > 0] = 1
    sum_Vr = np.nansum(np.square(parameters.vrs * parameters.weights))
    parameters.rmsVr = np.sqrt(np.nansum(sum_Vr) / np.nansum(parameters.weights))

    del bca
    parameters.grid_shape = u_init.shape
    # Parse names of velocity field

    winds = winds.flatten()

    print("Starting solver ")
    parameters.dx = np.diff(Grids[0].x['data'], axis=0)[0]
    parameters.dy = np.diff(Grids[0].y['data'], axis=0)[0]
    parameters.dz = np.diff(Grids[0].z['data'], axis=0)[0]
    print('rmsVR = ' + str(parameters.rmsVr))
    print('Total points: %d' % parameters.weights.sum())
    parameters.z = Grids[0].point_z['data']
    parameters.x = Grids[0].point_x['data']
    parameters.y = Grids[0].point_y['data']
    bt = time.time()

    # First pass - no filter
    wprevmax = 99
    wcurrmax = w_init.max()
    print("The max of w_init is", wcurrmax)
    iterations = 0
    bounds = [(-x, x) for x in 100*np.ones(winds.shape)]

    if(model_fields is not None):
        for the_field in model_fields:
            u_field = ("U_" + the_field)
            v_field = ("V_" + the_field)
            w_field = ("W_" + the_field)
            parameters.u_model.append(Grids[0].fields[u_field]["data"])
            parameters.v_model.append(Grids[0].fields[v_field]["data"])
            parameters.w_model.append(Grids[0].fields[w_field]["data"])

    parameters.Co = Co
    parameters.Cm = Cm
    parameters.Cx = Cx
    parameters.Cy = Cy
    parameters.Cz = Cz
    parameters.Cb = Cb
    parameters.Cv = Cv
    parameters.Cmod = Cmod
    parameters.Cpoint = Cpoint
    parameters.roi = roi
    parameters.upper_bc = upper_bc
    parameters.points = points
    parameters.point_list = points
    parameters.print_out = False


    params = copy.copy(parameters)
    fun = lambda z, prec: obj_fun(z, params, prec, jax_on=jax_on)

    t0 = time.time()
    temp = list()

    # are we using trophy?
    if use_dynTR:
        delta_init = norm(grad_J(winds, parameters, 'double', jax_on=False))

        print('Using gtol:', gtol, 'Memory:', max_memory)

        ret = DynPrec.DynTR(winds, fun, precision_dict, gtol=gtol, max_iter=max_iterations, verbose=True,
                                      max_memory=max_memory, delta_init=delta_init, store_history=True,
                                      tr_tol=subproblem_tol, write_folder=write_folder, sr1_tol=1.e-4)
        t_elapsed = time.time() - t0

        # the following allows us to go between stucture type to tuple type returned by fmin_l_bfgs_b
        winds = tuple((ret.x, ret.fun))
        temp = [ret.success, t_elapsed, np.float64(ret.fun), norm(ret.jac), norm(ret.jac, np.inf),
                     ret.nit, ret.nfev, ret.message]
        for key in ret.precision_counts:
            temp.append(ret.precision_counts[key])
        for key in ret.precision_counts:
            temp.append(ret.time_counter[key])
        temp = [temp, temp]  # pain in the butt work-around to save time time trying to write to csv
    else:
        new_J_function = lambda z, param: J_function(z, param, 'double')
        new_grad_J = lambda z, param: grad_J(z, param, 'double')
        winds = fmin_l_bfgs_b(new_J_function, winds, args=(parameters,), m=max_memory, maxiter=max_iterations,
                              pgtol=gtol, bounds=bounds, fprime=new_grad_J, iprint=99, factr=1e-32)
        t_elapsed = time.time() - t0
        # last arg ensure doesn't stop until max it or small gradient.
        my_f = winds[1]
        d = winds[2]

        successful = True if "NORM_OF_PROJECTED" in d['task'] else False
        temp.append([successful, t_elapsed, my_f, norm(d['grad']), norm(d['grad'], np.inf),
                     d['nit'], d['funcalls'], 0, d['task']])

    print("Done! Time = " + "{:2.1f}".format(time.time() - bt))

    ### WRITE RESULTS TO FILE IF STORE_HISTORY IS TRUE###
    if store_history:
        print('Writing files to ' + write_folder)
        fields = ['success', 'time', 'feval', 'grad2norm',
                  'grad_inf_norm', 'nits', 'fevals', 'message']
        for key in ret.precision_counts:
            fields.append(key + '_counts')
        for key in ret.time_counter:
            fields.append(key + '_tottime')

        df_general = pd.DataFrame(data=temp, columns=fields)
        df_general.drop(index=1, inplace=True)  # drop additional unecessary row
        df_general.to_csv(write_folder + '/summary.csv', index=False)


        if use_dynTR:
            histories = np.column_stack((ret.prec_hist, ret.f_hist, ret.g_two_norm_hist, ret.g_inf_norm_hist))
            df_func_grad_hist = pd.DataFrame(data=histories, columns=['precision', 'f_val', 'g_two_norm', 'g_inf_norm'])
            df_func_grad_hist.to_csv(write_folder + 'func_grad_hist.csv')
            x_and_grad = np.column_stack((ret.x, ret.jac))
            histories = np.column_stack((ret.it_prec_hist, ret.it_f_hist, ret.it_g_two_norm_hist,
                                         ret.it_g_inf_norm_hist, ret.it_step_size_hist,
                                         ret.it_delta_hist, ret.it_criteria_hist))

            df_func_grad_hist = pd.DataFrame(data=histories, columns=['precision', 'f_val', 'g_two_norm',
                                                                      'g_inf_norm', 'step_size', 'delta', 'criteria'])
            df_func_grad_hist.to_csv(write_folder + 'hist_by_iteration.csv')

        else:
            gtemp = grad_J(winds[0], parameters)
            x_and_grad = np.column_stack((winds[0], gtemp))

        df_x_and_grad = pd.DataFrame(data=x_and_grad, columns=['winds', 'gradient'])
        df_x_and_grad.to_csv(write_folder + 'winds_and_gradient.csv')


    winds = np.reshape( winds[0], (3, parameters.grid_shape[0], parameters.grid_shape[1], parameters.grid_shape[2]))
    iterations = iterations+10
    print('Iterations before filter: ' + str(iterations))
    winds = np.stack([winds[0], winds[1], winds[2]])
    winds = winds.flatten()




    if filt_iterations > 0:
        print('Applying low pass filter to wind field...')
        winds = np.reshape(winds, (3, parameters.grid_shape[0], parameters.grid_shape[1],
                                   parameters.grid_shape[2]))
        winds[0] = savgol_filter(winds[0], filter_window, filter_order, axis=0)
        winds[0] = savgol_filter(winds[0], filter_window, filter_order, axis=1)
        winds[0] = savgol_filter(winds[0], filter_window, filter_order, axis=2)
        winds[1] = savgol_filter(winds[1], filter_window, filter_order, axis=0)
        winds[1] = savgol_filter(winds[1], filter_window, filter_order, axis=1)
        winds[1] = savgol_filter(winds[1], filter_window, filter_order, axis=2)
        winds[2] = savgol_filter(winds[2], filter_window, filter_order, axis=0)
        winds[2] = savgol_filter(winds[2], filter_window, filter_order, axis=1)
        winds[2] = savgol_filter(winds[2], filter_window, filter_order, axis=2)
        winds = np.stack([winds[0], winds[1], winds[2]])
        winds = winds.flatten()
        iterations = 0
        while iterations < filt_iterations:
            # Supposed to smooth functions here
            filt_max_it = 10
            if use_dynTR:
                ret = DynPrec.DynTR_for_pydda(winds, fun, prec_vec, gtol=gtol, max_iter=filt_max_it, tr_tol=gtol, verbose=True,
                                    max_memory=max_memory, delta_init=delta_init)
                winds=tuple((ret.x, ret.fun))
            else:
                winds = fmin_l_bfgs_b(J_function, winds, args=(parameters,), m=max_memory,
                                      maxiter=filt_max_it, pgtol=gtol, bounds=bounds,
                                      fprime=grad_J, iprint=-1)


            parameters.print_out = False
            winds = np.reshape(
                winds[0], (3, parameters.grid_shape[0], parameters.grid_shape[1], parameters.grid_shape[2]))
            iterations = iterations+1
            print('Iterations after filter: ' + str(iterations))
            winds = np.stack([winds[0], winds[1], winds[2]])
            winds = winds.flatten()


    give_time = os.system('date')
    the_winds = np.reshape(
        winds, (3, parameters.grid_shape[0], parameters.grid_shape[1], parameters.grid_shape[2]))
    u = the_winds[0]
    v = the_winds[1]
    w = the_winds[2]
    where_mask = np.sum(parameters.weights, axis=0) + \
                 np.sum(parameters.model_weights, axis=0)
    
    u = np.ma.array(u)
    w = np.ma.array(w)
    v = np.ma.array(v)

    if mask_outside_opt is True:
        u = np.ma.masked_where(where_mask < 1, u)
        v = np.ma.masked_where(where_mask < 1, v)
        w = np.ma.masked_where(where_mask < 1, w)

    if mask_w_outside_opt is True:
        w = np.ma.masked_where(where_mask < 1, w)

    u_field = deepcopy(Grids[0].fields[vel_name])
    u_field['data'] = u
    u_field['standard_name'] = 'u_wind'
    u_field['long_name'] = 'meridional component of wind velocity'
    u_field['min_bca'] = min_bca
    u_field['max_bca'] = max_bca
    v_field = deepcopy(Grids[0].fields[vel_name])
    v_field['data'] = v
    v_field['standard_name'] = 'v_wind'
    v_field['long_name'] = 'zonal component of wind velocity'
    v_field['min_bca'] = min_bca
    v_field['max_bca'] = max_bca
    w_field = deepcopy(Grids[0].fields[vel_name])
    w_field['data'] = w
    w_field['standard_name'] = 'w_wind'
    w_field['long_name'] = 'vertical component of wind velocity'
    w_field['min_bca'] = min_bca
    w_field['max_bca'] = max_bca

    new_grid_list = []

    for grid in Grids:
        temp_grid = deepcopy(grid)
        temp_grid.add_field('u', u_field, replace_existing=True)
        temp_grid.add_field('v', v_field, replace_existing=True)
        temp_grid.add_field('w', w_field, replace_existing=True)
        new_grid_list.append(temp_grid)

    return new_grid_list


def get_bca(rad1_lon, rad1_lat, rad2_lon, rad2_lat, x, y, projparams):
    """
    This function gets the beam crossing angle between two lat/lon pairs.

    Parameters
    ==========
    rad1_lon: float
        The longitude of the first radar.
    rad1_lat: float
        The latitude of the first radar.
    rad2_lon: float
        The longitude of the second radar.
    rad2_lat: float
        The latitude of the second radar.
    x: nD float array
        The Cartesian x coordinates of the grid
    y: nD float array
        The Cartesian y corrdinates of the grid
    projparams: Py-ART projparams
        The projection parameters of the Grid

    Returns
    =======
    bca: nD float array
        The beam crossing angle between the two radars in radians.

    """

    rad1 = pyart.core.geographic_to_cartesian(rad1_lon, rad1_lat, projparams)
    rad2 = pyart.core.geographic_to_cartesian(rad2_lon, rad2_lat, projparams)
    # Create grid with Radar 1 in center

    x = x-rad1[0]
    y = y-rad1[1]
    rad2 = np.array(rad2) - np.array(rad1)
    a = np.sqrt(np.multiply(x, x) + np.multiply(y, y))
    b = np.sqrt(pow(x-rad2[0], 2) + pow(y-rad2[1], 2))
    c = np.sqrt(rad2[0]*rad2[0] + rad2[1]*rad2[1])
    theta_1 = np.arccos(x/a)
    theta_2 = np.arccos((x-rad2[1])/b)
    return np.arccos((a*a+b*b-c*c)/(2*a*b))
