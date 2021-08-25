Config = {
    # -----------------------------------------------------------------#
    #   always remember to change 'num_classes'
    # -----------------------------------------------------------------#
    'num_classes': 5,
    # -----------------------------------------------------------------#
    #   min_dim has two options, i.e., 300 and 512.
    #   The SSD512 here is not the original SSD512.
    #   The original SSD512 has one more prediction layer than SSD300;
    #   It is more troublesome to modify, so I just modified the input size
    #   This can also be used for training with larger images, which is good for small targets
    #   when min_dim = 512,'feature_maps': [64, 32, 16, 8, 6, 4]
    #   when min_dim = 300, 'feature_maps': [38, 19, 10, 5, 3, 1]
    # -----------------------------------------------------------------#
    'min_dim': 300,
    'feature_maps': {
        'vgg': [38, 19, 10, 5, 3, 1],
        'mobilenet': [19, 10, 5, 3, 2, 1],
    },
    # 'min_dim': 512,
    # 'feature_maps': {
    #     'vgg'       : [64, 32, 16, 8, 6, 4],
    #     'mobilenet' : [32, 16, 8, 4, 2, 1],
    # },

    # ----------------------------------------------------#
    #   min_sizes and max_sizes can be used to set the size of prior boxes
    #   The default is set according to the voc data set, which is common in most cases!
    #   If you want to detect small objects, you can modify
    #   Generally, just reduce the size of the shallow a priori box!
    #   Because the shallow layer is responsible for small object detection!
    #   e.g., min_sizes = [21,45,99,153,207,261]
    #         max_sizes = [45,99,153,207,261,315]
    # ----------------------------------------------------#
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],

    'aspect_ratios': {
        'vgg': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        'mobilenet': [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]]
    },
    'variance': [0.1, 0.2],
    'clip': True,
}