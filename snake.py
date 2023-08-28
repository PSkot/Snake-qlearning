import pygame
import numpy as np
import random
from qlearning import QLearning


def handle_input(action, dir_x, dir_y):
    if action == 0 and not dir_y:
        return (0, -1)
    if action == 2 and not dir_y:
        return (0, 1)
    if action == 3 and not dir_x:
        return (-1, 0)
    if action == 1 and not dir_x:
        return(1, 0)
    return (dir_x, dir_y)

class map:
    def __init__(self, width, height, snake):
        self.width = width
        self.height = height
        self.indices = [[i, j] for i in range(width) for j in range(width)]
        self.spawn_candy(snake)

    def spawn_candy(self, snake):
        # Spawn candy at a random location in the grid that the snake does not occupy
        choice = random.choice([i for i in self.indices if i not in snake])
        self.candy_x, self.candy_y = choice

class snake:
    def __init__(self, width, height):
        self.reset(width, height)

    def reset(self, width, height):
        init_x, init_y = (width//2, height//2)
        self.snake = [[init_x, init_y], [init_x - 1, init_y], [init_x - 2, init_y]]
        self.rounds = 200
        self.dir_x = 1
        self.dir_y = 0

    def update(self, map, width, height, action):
        # Handle input
        self.dir_x, self.dir_y = handle_input(action, self.dir_x, self.dir_y)

        # Move snake
        self.snake = [[self.snake[0][0] + self.dir_x, self.snake[0][1] + self.dir_y]] + self.snake[:-1]

        # Wall collision
        if self.snake[0][0] < 0 or self.snake[0][1] < 0 or self.snake[0][0] > width-1 or self.snake[0][1] > height-1:
            self.reset(width, height)
            map.spawn_candy(self.snake)
            return -100, True

        for i, s in enumerate(self.snake):
            # Candy collision
            if map.candy_x == self.snake[0][0] and map.candy_y == self.snake[0][1]:
                map.spawn_candy(self.snake)
                self.grow()
                self.rounds += 50
                return 200, False

            # Collision with self
            if i > 0 and s[1] == self.snake[0][1] and s[0] == self.snake[0][0]:
                self.reset(width, height)
                map.spawn_candy(self.snake)
                return -100, True

        self.rounds -= 1
        if self.rounds <= 0:
            return 0, True

        return -0, False

    def grow(self):
        self.snake.append([self.snake[0][0] - self.dir_x, self.snake[0][1] - self.dir_y])

class game:
    def __init__(self, width, height, fps = 20, tilesize = 10, learner = None):
        self.width = width
        self.height = height
        self.fps = fps
        self.tilesize = tilesize
        self.events = []
        self.snake = snake(width, height)
        self.map = map(width, height, self.snake.snake)
        self.learner = learner
        self.n_actions = 5

    def init_game(self):
        self.running = True
        if self.render:
            self.clock = pygame.time.Clock()
            self.screen = pygame.display.set_mode((self.width*self.tilesize, self.height*self.tilesize))

    def update(self, action):
        if self.render:
            self.clock.tick(self.fps)
            self.events = pygame.event.get()

            for event in self.events:
                if event.type == pygame.QUIT:
                    self.running = False

        return self.snake.update(self.map, self.width, self.height, action)

    def run(self, render, QTable = None):
        self.render = render
        self.init_game()
        if QTable is not None:
            self.learner.Q = QTable

        while self.running:
            state, action, reward, done = self.compute_reward()
            self.learner.update(state, action, reward)
            if self.render:
                self.render_game()

    def train(self, epochs = 10000, model = None):
        if model is not None:
            self.learner.Q = model
        self.render = False
        e = 0
        self.init_game()
        while self.running:
            state, action, reward, done = self.compute_reward()
            self.learner.update(state, action, reward)
            if done:
                e += 1
                if e%1000 == 0:
                    np.save('QTable.npy', self.learner.Q)
                    print("Epoch:", e)
            if e == epochs:
                self.running = False


    def compute_reward(self):
        state = self.get_state()
        old_x_dist, old_y_dist = abs(self.snake.snake[0][0] - self.map.candy_x), abs(self.snake.snake[0][1] - self.map.candy_y)
        action = self.learner.step(state, table_only = True)
        reward, done = self.update(action)
        if not reward and not done:
            new_x_dist, new_y_dist = abs(self.snake.snake[0][0] - self.map.candy_x), abs(self.snake.snake[0][1] - self.map.candy_y)
            if new_x_dist > old_x_dist:
                reward = -1
            elif new_y_dist > old_y_dist:
                reward = -1
            else:
                reward = 0.5

        return state, action, reward, done

    def get_state(self):
        # Get direction index
        if self.snake.dir_x > 0:
            i_dir = 1
        elif self.snake.dir_x < 0:
            i_dir = 3
        elif self.snake.dir_y > 0:
            i_dir = 2
        else:
            i_dir = 0

        # Get xaxis index
        if self.snake.snake[0][0] > self.map.candy_x:
            i_lr = 2
        elif self.snake.snake[0][0] < self.map.candy_x:
            i_lr = 0
        else:
            i_lr = 1

        # Get yaxis index
        if self.snake.snake[0][1] > self.map.candy_y:
            i_ab = 2
        elif self.snake.snake[0][1] < self.map.candy_y:
            i_ab = 0
        else:
            i_ab = 1

        # Next to left wall
        i_lw = self.snake.snake[0][0] == 0

        # Next to right wall
        i_rw = self.snake.snake[0][0] == self.width-1

        # Next to top wall
        i_tw = self.snake.snake[0][1] == 0

        # Next to bottom wall
        i_bw = self.snake.snake[0][1] == self.height-1

        return i_dir*144+i_lr*48+i_ab*16+i_lw*8+i_rw*4+i_tw*2+i_bw

    def render_game(self):
        self.screen.fill((0, 0, 0))
        pygame.draw.rect(self.screen, (200, 200, 200), (self.map.candy_x*self.tilesize, self.map.candy_y*self.tilesize, self.tilesize, self.tilesize))

        for s in self.snake.snake:
            pygame.draw.rect(self.screen, (150, 0, 0), (s[0]*self.tilesize, s[1]*self.tilesize, self.tilesize, self.tilesize))

        pygame.display.update()

if __name__ == '__main__':
    learner = QLearning(q_shape = (144*4, 5))
    snek = game(15, 15, learner = learner)
    # snek.train(epochs = 5000)

    QTable = np.load('QTable.npy')
    snek.run(render = True, QTable = QTable)
