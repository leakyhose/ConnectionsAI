"""
Generates visual representation of the Connections AI
"""

import pygame
import sys
import time
import numpy as np
import pygame_gui
import copy
from game_master import check, push, pop, genPq, linkPq, childPq, purge


pygame.init()

WIDTH, HEIGHT = 1600, 900
GRID_WIDTH, GRID_HEIGHT = 600, 600
ROWS, COLS = 4, 4
CELL_WIDTH, CELL_HEIGHT = GRID_WIDTH // COLS, GRID_HEIGHT // ROWS

MANAGER = pygame_gui.UIManager((WIDTH, HEIGHT))
CLOCK = pygame.time.Clock()

TEXT_INPUT = pygame_gui.elements.UITextEntryLine(
    relative_rect=pygame.Rect((200, 60), (555, 50)),
    manager=MANAGER,
    object_id="#main_text_entry",
)

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Connections Simulator Bot")


font = pygame.font.SysFont("sitkabanner", 22)


grid_x = (WIDTH - GRID_WIDTH) // 5
grid_y = (HEIGHT - GRID_HEIGHT) // 2


def truncate(value):
    """
    Truncate a floating-point number to 3 decimal places.
    """
    return f"{value:.3f}"


def draw_grid(words, avail):
    """
    Draw a grid of words on the screen, using available indices from 'avail'.
    """
    words = [words[idx] for idx in avail if 0 <= idx < len(words)]
    np.random.shuffle(words)

    ROW = int(len(words) / 4)
    COL = 4

    for row in range(ROW):
        for col in range(COL):

            x = grid_x + col * CELL_WIDTH
            y = grid_y + row * CELL_HEIGHT
            pygame.draw.rect(screen, WHITE, (x, y, CELL_WIDTH, CELL_HEIGHT), 1)
            word = words[row * COL + col]
            text = font.render(word, True, WHITE)
            text_rect = text.get_rect(
                center=(x + CELL_WIDTH // 2, y + CELL_HEIGHT // 2)
            )
            screen.blit(text, text_rect)
            pygame.display.update()
            time.sleep(0.02)


def remove_indices(words, indices):
    """
    Remove elements from 'words' based on the provided 'indices'.
    """
    indices_to_remove = set(indices)
    result = [word for idx, word in enumerate(words) if idx not in indices_to_remove]
    return result


def convertIndex(index, arr):
    """
    Convert the indices in 'arr' to strings using the 'index' list.
    """
    for i in range(len(arr)):
        arr[i] = str(index[arr[i]])
    return arr


def draw_pq(index, queue):
    """
    Draw the priority queue on the screen with the parsed results.
    """
    i = 0
    y_offset = 70

    while len(queue) > 0:

        if i >= 20:
            break

        out = pop(queue)
        text = font.render(
            f"{i+1}: {parse_result(convertIndex(index, list(out[0])))} - {truncate(out[1])}",
            True,
            "WHITE",
        )
        screen.blit(text, (850, y_offset))

        y_offset += 35

        i += 1

        pygame.display.update()
        time.sleep(0.02)


def display_result(out, curr, index):
    """
    Display the result of the user's selection (correct, incorrect, or one away).
    """
    curr = parse_result(convertIndex(index, curr))

    if out == -1:
        show_text(str(curr) + " is incorrect.")

    elif out == 0:
        show_text(str(curr) + " is one away.")

    else:
        show_text(str(curr) + " is correct!")


def parse_result(result):
    """
    Parse and format the result array into a comma-separated string.
    """
    s = ""

    for word in result:
        word.lower()
        s += word
        s += ", "

    return s[:-2]


def show_text(word):
    """
    Display a single line of text in the center of the screen and wait.
    """
    word = str(word)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        screen.fill(BLACK)

        text = font.render(word, True, WHITE)
        text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        screen.blit(text, text_rect)
        pygame.display.update()

        time.sleep(1)

        screen.fill(BLACK)

        return


def pop_specific(pq, n):
    """
    Pop the 'n'th element from the priority queue.
    """
    temp = []

    for i in range(n - 1):
        out = pop(pq)
        push(temp, out[0], out[1])

    out = pop(pq)

    while len(temp) > 0:
        popped = pop(temp)
        push(pq, popped[0], popped[1])

    pq += temp

    return (out, pq)


def user_pop(words, pq, avail):
    """
    Allow the user to pop an element from the priority queue based on input.
    """
    shuffled = copy.deepcopy(words)

    screen.fill(BLACK)
    draw_grid(shuffled, avail)
    draw_pq(words, pq[:])

    while True:

        UI_REFRESH_RATE = CLOCK.tick(60) / 1000

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if (
                event.type == pygame_gui.UI_TEXT_ENTRY_FINISHED
                and event.ui_object_id == "#main_text_entry"
            ):

                input = event.text

                if int(input) <= min(len(pq), 20) and int(input) > 0:
                    temp = pop_specific(pq, int(input))
                    return temp[0][0], temp[1]

            MANAGER.process_events(event)

        MANAGER.update(UI_REFRESH_RATE)

        MANAGER.draw_ui(screen)

        pygame.display.update()



def play(n, data_name, weights):
    """
    The main game loop, playing the nth connection game in "data_name" folder with the provided weights.
    """
    word_archive = np.load(data_name + "/word_data.npy")
    words = word_archive[n]
    adj = np.load(data_name + "/" + "data.npy", allow_pickle=True)[n]
    avail = range(16)
    pq = genPq(adj, avail, weights)

    turns = 0

    while len(avail) != 0:

        out = -1
        curr = []
        pq = genPq(adj, avail, weights)

        while out == -1:

            temp = user_pop(words, pq[:], avail)

            curr = temp[0]
            pq = temp[1]
            out = check(curr)

            display_result(out, curr[:], words)

            turns += 1

        if out == 0:

            trios = linkPq(curr, adj, weights)
            out = -1

            while out != 1:

                bestTrio = pop(trios)[0]
                pq = childPq(bestTrio, adj, avail, weights)

                temp = user_pop(words, pq[:], avail)
                curr = temp[0]
                pq = temp[1]
                out = check(curr)

                display_result(out, curr[:], words)

                turns += 1

                while out == 0:

                    temp = user_pop(words, pq[:], avail)
                    curr = temp[0]
                    pq = temp[1]
                    out = check(curr)

                    display_result(out, curr[:], words)

                    turns += 1

            else:
                adj = purge(adj, curr)
                avail = list(set(avail) - set(curr))

        else:
            adj = purge(adj, curr)
            avail = list(set(avail) - set(curr))

    show_text("Amount of Mistakes: " + str(turns - 4))
    return turns

WEIGHTS = (0.7441864013671875, 0.06005859375)

play(1, "fasttext", WEIGHTS)
