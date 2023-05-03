import random
import pygame
import time
import cv2

##########   General   ##########
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

FLIP = True
FONTSIZE = 40
COLOR_TEXT = WHITE
DISPSIZE = (1920, 1680)
COLOR_BACKGROUND = WHITE

def cap(cam):
    _, frame = cam.read()
    if FLIP:
        frame = cv2.flip(frame, 0)

    return frame


def wait_space_key():
    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                return


def display_instruction(inst, disp, dispsize, wait_time):
    disp.fill(BLACK)
    # TODO: generalize this section.
    x = dispsize[0] / 2
    y = dispsize[1] / 2 - 50

    font = pygame.font.Font(None, FONTSIZE)
    rendered = font.render(inst, True, COLOR_TEXT)
    disp.blit(rendered, (x, y))

    pygame.display.update()
    time.sleep(wait_time)


def collect_four_corner_points(sample_num, disp, corner_lst, size_list, camera):
    str_list_tags = ["A", "B", "C", "D"]

    # makes sure the square are actually corner markers
    x_offsets = [0, size_list[0]-25, 0, size_list[0]-25]
    y_offsets = [0, 0, size_list[1]-25, size_list[1]-25]
    for x in range(len(corner_lst)):
        disp.fill(COLOR_BACKGROUND)
        point = corner_lst[x]
        tag = str_list_tags[x]

        # TODO: Should these be segments or staple points for the corner?
        mark = pygame.Rect(point[0]+x_offsets[x], point[1]+y_offsets[x], 25, 25)
        pygame.draw.rect(disp, BLACK, mark)
        pygame.display.update()

        file_name = "data/{}_calib_{}.jpg".format(sample_num, tag)

        wait_space_key()
        frame = cap(camera)
        cv2.imwrite(file_name, frame)


def collect_four_rand_points(sample_num, num_vert, num_hor, num_to_collect, disp, size_lst, camera):
    for x in range(num_to_collect):
        disp.fill(COLOR_BACKGROUND)
        CUBE_COLOR = (25*x%150+100, 25*x%150, 25*x%255)

        # Choose random (x, y) coord for segment
        x_coord = random.randint(0, num_hor-1)
        y_coord = random.randint(0, num_vert-1)

        segment_coord = (x_coord * size_lst[0], y_coord * size_lst[1])

        # Highlight segment
        mark = pygame.Rect(segment_coord[0], segment_coord[1], size_lst[0], size_lst[1])
        pygame.draw.rect(disp, CUBE_COLOR, mark)

        font = pygame.font.Font(None, FONTSIZE)
        rendered = font.render("HERE", True, WHITE)
        disp.blit(rendered, (segment_coord[0] + size_lst[0]/2, segment_coord[1] + size_lst[1]/2 - 25))

        pygame.display.update()

        # Record user's gaze when space pressed
        file_name = "data/{}_data_{}_{}_{}.jpg".format(sample_num, x, x_coord, y_coord)
        wait_space_key()
        frame = cap(camera)

        cv2.imwrite(file_name, frame)


def collect_data():
    # MOVE THESE TO CONSTANTS SECTION
    num_horizontal_classifications = 4
    num_vertical_classifications = 4
    number_of_samples_to_take = 12
    num_data_per_calib = 10

    # Define size of screen
    display = pygame.display.get_surface()
    dispsize = pygame.display.get_surface().get_size()

    segment_size_width = dispsize[0] / num_horizontal_classifications
    segment_size_height = dispsize[1] / num_vertical_classifications

    # Prepare camera capture
    cap = cv2.VideoCapture(0)
    # TODO: Check if this is correct?
    cap.set(5, 60)

    #   Representation of screen key points (ODD)
    #  TL           TM          TR
    #
    #  ML           MM          MR
    #
    #  BL           BM          BR
    #
    # Where
    #   T: Top
    #   M: Mid
    #   B: Bottom
    # And TL == (0,0)       BR == (display_width, display_height)
    # All points indicate the top left corner of the segment
    # TODO: For now, only TL, TR, BL, BR used.

    TL = (0, 0)
    TR = (dispsize[0] - segment_size_width, 0)
    BL = (0, dispsize[1] - segment_size_height)
    BR = (dispsize[0] - segment_size_width, dispsize[1] - segment_size_height)
    corner_list = [TL, TR, BL, BR]
    size_list = [segment_size_width, segment_size_height]

    for i in range(0, number_of_samples_to_take):
        display_instruction("NEW SAMPLESET: {}".format(i), display, dispsize, 5)
        collect_four_corner_points(i, display, corner_list, size_list, cap)
        collect_four_rand_points(i,
                                 num_vertical_classifications,
                                 num_horizontal_classifications,
                                 num_data_per_calib,
                                 display,
                                 size_list,
                                 cap)


if __name__ == "__main__":
    pygame.init()

    dispsize = DISPSIZE
    # Launch Fullscreen mode
    display = pygame.display.set_mode(dispsize, pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF)
    collect_data()
    pygame.display.quit()
    pygame.quit()
