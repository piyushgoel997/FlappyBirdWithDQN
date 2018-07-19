import pygame
import random

print("importing done")

SCREEN_WIDTH = 422
SCREEN_HEIGHT = 422
BIRD_HEIGHT = 24
BIRD_WIDTH = 34
PIPE_HEIGHT = 300
PIPE_GAP = 100
PIPE_WIDTH = 75
FALL_ACCELERATION = 1
BIRD_MAX_VELOCITY = -10
min_valy = 8
PIPE_VELOCITY = 4

FPS = 30

pygame.init()
CLOCK = pygame.time.Clock()
game_screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Flappy Bird')

BIRD_IMG = pygame.image.load('Assets/bird.png')
BACKGROUND = pygame.image.load('Assets/background.png')
PIPE_IMG = pygame.image.load('Assets/pipe.png')


class Game:

    def __init__(self):
        self.birdx = int(0.2 * SCREEN_WIDTH)
        self.birdy = int(0.5 * SCREEN_HEIGHT)
        self.bird_velocity = 0

        self.score = 0
        self.pipes = []
        self.game_over = False

    def game_step(self, action):
        """
        takes one step through the game.
        :param action: 1-d list of size 2.
                       action[0] = 1 -> do nothing
                       action[1] = 1 -> jump
        """
        pygame.event.pump()

        reward = 0.1  # reward for the current step

        if action[1] == 1:
            self.bird_velocity = BIRD_MAX_VELOCITY

        # check if new pipe is needed
        pipe_needed = False
        for p in self.pipes:
            # if a pipe is at the mid, create one more. +10 because a pip is started at 10 from the right of the screen.
            if 10 + SCREEN_WIDTH/2 <= p['x'] + PIPE_WIDTH/2 < 10 + PIPE_VELOCITY + SCREEN_WIDTH/2:
                pipe_needed = True

        # create a new pipe
        if (len(self.pipes) is 0) or pipe_needed:
            for p in getRandomPipe():
                self.pipes.append(p)

        # destroy pipe if it goes off screen
        for p in self.pipes:
            if p['x'] + PIPE_WIDTH < 0:
                self.pipes.remove(p)
                break

        # check for score
        bird_mid = self.birdx + BIRD_WIDTH
        for p in self.pipes:
            p_mid = p['x'] + PIPE_WIDTH / 2
            if p_mid <= bird_mid < p_mid + PIPE_VELOCITY:
                reward = 1
                break  # the bird can only be crossing one set of pipes at a time

        if check_crashed(self.birdx, self.birdy, self.pipes):
            reward = -1
            self.game_over = True
            # TODO do something if game over

        # display everything
        game_screen.blit(BACKGROUND, (0, 0))
        game_screen.blit(BIRD_IMG, (self.birdx, self.birdy))
        display_pipes(self.pipes)

        # update coordinates
        self.birdy += self.bird_velocity
        self.bird_velocity += FALL_ACCELERATION
        for p in self.pipes:
            p['x'] -= PIPE_VELOCITY

        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()

        CLOCK.tick(FPS)

        return image_data, reward, self.game_over


def getRandomPipe():
    gapup = random.randint(10, 27) * 10
    upx = SCREEN_WIDTH + 10
    lowx = SCREEN_WIDTH + 10
    upy = gapup - PIPE_HEIGHT
    lowy = gapup + PIPE_GAP

    return [
        {'x': upx, 'y': upy},
        {'x': lowx, 'y': lowy}
    ]


def check_crashed(birdx, birdy, pipes):
    for p in pipes:
        pxl = p['x']
        pxr = p['x'] + PIPE_WIDTH
        pyu = p['y']
        pyd = p['y'] + PIPE_HEIGHT

        if pxl <= birdx + BIRD_WIDTH <= pxr or pxl <= birdx <= pxr:
            if pyu <= birdy + BIRD_HEIGHT <= pyd or pyu <= birdy <= pyd:
                return True

    if birdy <= 0:
        return True

    if birdy + BIRD_HEIGHT >= SCREEN_HEIGHT:
        return True

    return False


def display_pipes(pipes):
    for p in pipes:
        game_screen.blit(PIPE_IMG, (p['x'], p['y']))
        # pygame.draw.rect(game_screen, (0, 0, 0), [p['x'], max(0, p['y']),
        #                                           p['x'] + PIPE_WIDTH, min(SCREEN_HEIGHT, p['y'] + PIPE_HEIGHT)])
    return


if __name__ == '__main__':
    game = Game()

    game_over = False

    while not game_over:
        action = [1, 0]
        for e in pygame.event.get():
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE:
                    action = [0, 1]
        _, rew, game_over = game.game_step(action=action)
