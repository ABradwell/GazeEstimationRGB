Game Layout:
    General pattern: Prepare Countdown information

    10 times:
        4 Calibration squares, one second for each
        4 classification squares, randomly
    30 second break


    top left: (0,0)
    bottom right: (dip_width, dip_height)


Gameplan for the project

    Get it to recognize the square being stared at, based on first four angles recognized.

    Steps:
        Have user look at all 4 corners, have them look at a random square on the grid of 8 x 8
        for each data point, there will be collected...
            1. 4 images where the user looks at each corner
            2. 4 images where the user is looking at some random square (as shown in image title)

                FORMAT FOR IMAGE TITLE:
                    CORNER_CAL_#_[A/B/C/D]
                    SAMPLE_#_[1/2/3/4/ etc]

                           where # is the number of datapoints taken.

            3. The network will then be trained to take 4 initial calibration points,
            and for each frame received estimate which square is being viewed.

    Repeat this training process for...
        8X8
        16X16
        32X32

    TODO: There needs to be more data collected on the face. We need to record all data we can such as
        1. Angle of face plane
        2. Width of face
        3. Height of Face
        4. Distance between eyes

        This increases the odds of predicting using moving heads