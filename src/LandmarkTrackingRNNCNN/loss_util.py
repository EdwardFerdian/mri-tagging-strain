import tensorflow as tf

def calculate_loss(labels, predictions, time_steps, strain_err_weight=5.):
    """
        Overall loss: MSE + w * error_rr + w * error_cc

        Return mean of combined loss, mean of rr error, mean of cc error
    """
    # mean squared error
    sqrdiff2 = tf.square(predictions - labels)
    sqrdiff2 = tf.reduce_mean(sqrdiff2, axis=2)

    # absolute radial strain error
    rr_diff = get_radial_strains(predictions, time_steps) - get_radial_strains(labels, time_steps)
    rr_diff = tf.abs(rr_diff)

    # absolute midwall circumferential strain error
    cc_diff = get_midwall_cc_strains(predictions, time_steps) - get_midwall_cc_strains(labels, time_steps)
    cc_diff = tf.abs(cc_diff)

    # combined loss
    loss2 = sqrdiff2 + (strain_err_weight * rr_diff) + (strain_err_weight * cc_diff)
    loss2 = tf.reduce_mean(loss2)

    return loss2, tf.reduce_mean(rr_diff), tf.reduce_mean(cc_diff)

def get_radial_strains(flattened_coords, time_steps):
    epsilon = 0.0001
    # reshape to batch x time x 2 x 168 points (24 radials with 7 points each)
    coords_batch = tf.reshape(flattened_coords, [-1, time_steps, 2, 168])
    endo_batch = coords_batch[:,:,:, ::7]
    epi_batch = coords_batch[:,:,:, 6::7]

    diff = tf.square(epi_batch - endo_batch)
    summ = diff[:,:,0,:] + diff[:,:,1,:] # x2 + y2
    

    summ_arr = tf.unstack(summ, axis=1)
    summ_ed = summ[:,0,:] + epsilon

    res = []
    for t in range(0,len(summ_arr)):
        if t == 0:
            summ_t = summ_arr[t] + epsilon  
        else:
            summ_t = summ_arr[t]
        res.append(summ_t / summ_ed)

    divv = tf.stack(res, axis=1)

    # do the strain formula
    res = (divv - 1) / 2
    res = tf.reduce_mean(res, axis=2)
    return res

def get_midwall_cc_strains(flattened_coords, time_steps):
    epsilon = 0.0001
    # reshape to batch x time x 2 x 168 points (24 radials with 7 points)
    coords_batch = tf.reshape(flattened_coords, [-1, time_steps, 2, 168])
    midwall_points = coords_batch[:,:,:, 3::7]  # get point index 3 for every radial

    # we will have to calculate the strain between every points
    points_arr = tf.unstack(midwall_points, axis=3)

    # strain formula: ((l^2/L^2)-1) / 2  --> l^2 = x^2 + y^2
    # with x and y is the difference between x and y coords of 2 points
    ccs = []
    # the cc strain is circular, so we going through all of each indexes and back to point 0
    for r in range(0,len(points_arr)):
        # for the last point, calculate between point_r and point_0
        if r+1 == len(points_arr):
            cc_diff = tf.square(points_arr[r] - points_arr[0])
        else:
            cc_diff = tf.square(points_arr[r] - points_arr[r+1])

        # do the sum: x^2 + y^2
        cc_sum = cc_diff[:,:,0] + cc_diff[:,:,1]

        # we need to unstack it first
        ccsum_arr = tf.unstack(cc_sum, axis=1)

        res_arr = []
        # we are going to do the l^2 / L^2 here for every frame
        for t in range(0,len(ccsum_arr)):
            # divide by time 0
            if t == 0:
                cc_divv = (ccsum_arr[t] + epsilon) / (cc_sum[:,0] + epsilon)                 
            else:
                cc_divv = ccsum_arr[t] / (cc_sum[:,0] + epsilon)

            # do the strain formula
            res = (cc_divv - 1) / 2
            res_arr.append(res)
        partial_cc = tf.stack(res_arr, axis=1)
        # put the partial_cc in every time frame back together
        ccs.append(partial_cc)
    # stack the partial_cc for every links together
    stacked_ccs = tf.stack(ccs, axis=2)

    # calculate the mean cc for every time frame
    mid_cc = tf.reduce_mean(stacked_ccs, axis=2)
    return mid_cc