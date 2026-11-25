import numpy as np
import onnxruntime as ort
import gc

nn_size = (128, 128)
def normalize(image):
    max_val = np.max(image)
    min_val = np.min(image)
    ran = max_val - min_val
    return (image - min_val) / ran
def get_x_set(image, step_x):
    return_list = []
    if image.shape[0] == nn_size[1]:
        if (image.shape[1] - nn_size[0]) % step_x == 0:
            for i in range(0, image.shape[1] - nn_size[0], step_x):
                arr = image[:, i:i + nn_size[0]]
                return_list.append(arr)
        else:
            for i in range(0, image.shape[1] - nn_size[0], step_x):
                arr = image[:, i:(i + nn_size[0])]
                return_list.append(arr)
            arr = image[:, image.shape[1] - nn_size[0]:image.shape[1]]
            return_list.append(arr)
        return return_list
    else:
        return None

def get_y_slice(image, step_x, step_y):
    return_list = []
    a, b, c = image.shape
    if (image.shape[0] - nn_size[0]) % step_y == 0:
        for i in range(0, a - nn_size[1], step_y):
            set_list = get_x_set(image[i:i + nn_size[1], :], step_x)
            return_list += set_list
    else:
        for i in range(0, a - nn_size[1], step_y):
            set_list = get_x_set(image[i:i + nn_size[1], :], step_x)
            return_list += set_list
        set_list = get_x_set(image[image.shape[0] - nn_size[0]:image.shape[0], :], step_x)
        return_list += set_list
    return return_list

def predict_band(current, loaded_path, step=(8, 8), batch_size=32):
    train_data = get_y_slice(current, step[0], step[1])
    train = np.array(train_data)

    session = ort.InferenceSession(loaded_path)
    input_name = session.get_inputs()[0].name

    num_samples = train.shape[0]
    preds = []

    # Run inference in mini-batches to prevent memory overload
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch = train[start_idx:end_idx].astype(np.float32)

        # If your model expects NCHW, transpose the batch:
        # batch = np.transpose(batch, (0, 3, 1, 2))

        batch_pred = session.run(None, {input_name: batch})[0]
        preds.append(batch_pred)

    pred = np.concatenate(preds, axis=0)

    # Free up unused memory
    del train, train_data
    gc.collect()

    # CORRECTED CHANNEL EXTRACTION:
    # Use pred[:, 0, :, :] if your model is NCHW (Batch, Channels, Height, Width)
    # Use pred[:, :, :, 0] if your model is NHWC (Batch, Height, Width, Channels)
    if pred.ndim == 4 and pred.shape[1] == 2:  # NCHW check
        pred0 = pred[:, 0, :, :]
    else:  # NHWC check
        pred0 = pred[:, :, :, 0]

    a, b, _ = current.shape
    step_x = step[0]
    step_y = step[1]

    res_y = True if (a - nn_size[0]) % step_y == 0 else False
    res_x = True if (b - nn_size[1]) % step_x == 0 else False

    arr_add_shape = (((a - nn_size[0]) // step_y) + 2,) if (a - nn_size[0]) % step_y != 0 else ((a - nn_size[0]) // step_y,)
    arr_zeros = np.zeros((a, b) + arr_add_shape)
    counter = 0

    for i in range(0, a - nn_size[0], step_y):
        ret_arr_shape = (((b - nn_size[1]) // step_x) + 2,) if (b - nn_size[1]) % step_x != 0 else ((b - nn_size[1]) // step_x,)
        return_arr = np.zeros((nn_size[0], b) + ret_arr_shape)

        for j in range(0, b - nn_size[1], step_x):
            return_arr[:, j:j + nn_size[0], j // step_x] = pred0[counter]
            counter += 1

        if not res_x:
            return_arr[:, b - nn_size[0]:b, (j + step_x) // step_x] = pred0[counter]
            counter += 1

        arr_zeros[i:i + nn_size[0], :, i // step_y] = np.max(return_arr, axis=2)

    if not res_y:
        return_arr = np.zeros((nn_size[0], b) + ret_arr_shape)

        for j in range(0, b - nn_size[1], step_x):
            return_arr[:, j:j + nn_size[0], j // step_x] = pred0[counter]
            counter += 1

        if not res_x:
            return_arr[:, b - nn_size[0]:b, (j + step_x) // step_x] = pred0[counter]
            counter += 1

        arr_zeros[a - nn_size[0]:a, :, i // step_y] = np.max(return_arr, axis=2)
        counter += 1

    arr_zeros = np.max(arr_zeros, axis=2)
    return arr_zeros

def predict_band_deeplab(current, loaded_path, step=(8, 8), batch_size=32):
    train_data = get_y_slice(current, step[0], step[1])
    train = np.array(train_data)

    session = ort.InferenceSession(loaded_path)
    input_name = session.get_inputs()[0].name

    num_samples = train.shape[0]
    preds = []

    # Run inference in mini-batches to prevent memory overload
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch = train[start_idx:end_idx].astype(np.float32)

        # If your model expects NCHW, transpose the batch:
        batch = np.transpose(batch, (0, 3, 1, 2))

        batch_pred = session.run(None, {input_name: batch})[0]
        preds.append(batch_pred)

    pred = np.concatenate(preds, axis=0)

    # Free up unused memory
    del train, train_data
    gc.collect()

    # CORRECTED CHANNEL EXTRACTION:
    #Use pred[:, 0, :, :] if your model is NCHW (Batch, Channels, Height, Width)
    # Use pred[:, :, :, 0] if your model is NHWC (Batch, Height, Width, Channels)
    #if pred.ndim == 4 and pred.shape[1] == 2:  # NCHW check
    #    pred0 = pred[:, 0, :, :]
    #else:  # NHWC check
     #   pred0 = pred[:, :, :, 0]
    pred0 = pred[:, 0, :, :]
    a, b, _ = current.shape
    step_x = step[0]
    step_y = step[1]

    res_y = True if (a - nn_size[0]) % step_y == 0 else False
    res_x = True if (b - nn_size[1]) % step_x == 0 else False

    arr_add_shape = (((a - nn_size[0]) // step_y) + 2,) if (a - nn_size[0]) % step_y != 0 else ((a - nn_size[0]) // step_y,)
    arr_zeros = np.zeros((a, b) + arr_add_shape)
    counter = 0

    for i in range(0, a - nn_size[0], step_y):
        ret_arr_shape = (((b - nn_size[1]) // step_x) + 2,) if (b - nn_size[1]) % step_x != 0 else ((b - nn_size[1]) // step_x,)
        return_arr = np.zeros((nn_size[0], b) + ret_arr_shape)

        for j in range(0, b - nn_size[1], step_x):
            return_arr[:, j:j + nn_size[0], j // step_x] = pred0[counter]
            counter += 1

        if not res_x:
            return_arr[:, b - nn_size[0]:b, (j + step_x) // step_x] = pred0[counter]
            counter += 1

        arr_zeros[i:i + nn_size[0], :, i // step_y] = np.max(return_arr, axis=2)

    if not res_y:
        return_arr = np.zeros((nn_size[0], b) + ret_arr_shape)

        for j in range(0, b - nn_size[1], step_x):
            return_arr[:, j:j + nn_size[0], j // step_x] = pred0[counter]
            counter += 1

        if not res_x:
            return_arr[:, b - nn_size[0]:b, (j + step_x) // step_x] = pred0[counter]
            counter += 1

        arr_zeros[a - nn_size[0]:a, :, i // step_y] = np.max(return_arr, axis=2)
        counter += 1

    arr_zeros = np.max(arr_zeros, axis=2)
    return arr_zeros