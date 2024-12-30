from math import sqrt
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
"""
Assignment: Solve a Maze using Dijkstra's Algorithm

Objective:
In this assignment, you will implement Dijkstra's algorithm to find the shortest path in a maze represented as an RGB image. You will write custom data structures, including a PixelNode class and a priority queue, while ensuring efficient pathfinding.

Problem Description:
You are given an RGB image of a maze where:
- White pixels (all channels ≈ 255) represent valid paths.
- Non-white pixels (one or more channels < 255) represent obstacles.

Your goal is to find the shortest path between a start pixel and an end pixel, given their coordinates, by:
1. Converting the RGB maze image to a grayscale representation for simplicity.
2. Implementing a custom PixelNode class for maze pixels.
3. Building and utilizing a priority queue with a decrease_key operation for efficient pathfinding.
4. Using Dijkstra's algorithm to traverse the maze and determine the shortest path.

---

Tasks:

1. Grayscale Conversion
Write a function to convert the RGB maze image into grayscale using the formula:
  Gray = 0.2989 * R + 0.5870 * G + 0.1140 * B

Task:
Implement a function `convert_to_grayscale(image)` that:
- Takes the RGB image as input.
- Converts it to grayscale using the formula.
- Returns the grayscale image as a 2D array.

"""


def convert_to_grayscale(image):
    """
    Convert an RGB image to grayscale.

    :param image: Input RGB image as a 3D NumPy array
    :return: Grayscale image as a 2D NumPy array
    """
    for i, row in enumerate(image):
        for j, cell in enumerate(row):
            R,G,B = cell
            gray = 0.2989 * R + 0.5870 * G + 0.1140 * B
            image[i][j] = int(gray)
    
    return image
    


"""
2. Pixel Representation
Create a class `PixelNode` to represent each pixel in the maze. It should include:
- x and y: Coordinates of the pixel.
- distance: Shortest known distance from the start pixel (initialize to infinity).
- visited: Boolean indicating whether the pixel has been processed.
- color: Optional attribute for the grayscale intensity value.
- heap_index: The index of the PixelNode in the priority queue. This is essential for efficient decrease_key operations.

Additionally:
- Implement comparison operators (e.g., __lt__) to make PixelNode objects compatible with a min-heap.

"""


class PixelNode:
    def __init__(self, x, y, color=None):
        """
        Initialize a PixelNode object.

        :param x: X-coordinate of the pixel
        :param y: Y-coordinate of the pixel
        :param color: Grayscale intensity of the pixel
        """
        self.x = x
        self.y = y
        self.color = color
        self.distance = float('inf')
        self.visited = False
        self.heap_index = None
        self.prev = None

    def _distance(self, other):
        x1, y1 = self.x, self.y
        x2, y2 = other.x, other.y
        return sqrt((x1-x2)**2 + (y1-y2)**2)
        

    def __lt__(self, other):
        """
        Comparison operator for PixelNode based on distance.
        """
        return self.distance < other.distance
    
    def __gt__(self, other):
        """
        Comparison operator for PixelNode based on distance.
        """
        return self.distance > other.distance


"""
3. Priority Queue
Write a class `PriorityQueue` for managing PixelNode objects. Use a min-heap for efficient operations. Include the following methods:
- insert(node): Add a new PixelNode to the queue and update its heap_index.
- extract_min(): Remove and return the PixelNode with the smallest distance while maintaining the heap property.
- decrease_key(node, new_distance): Update the distance of an existing node and adjust its position in the heap. Update its heap_index.

Hint:
Use the heap_index attribute of PixelNode objects to keep track of their position in the heap. This allows decrease_key to access nodes efficiently.
"""

class PriorityQueue:
    @staticmethod
    def left_child(i):
        return (i*2)+1 
    
    @staticmethod
    def right_child(i):
        return (i*2)+2
    
    @staticmethod
    def father(i):
        return (i-1)//2
    

    def __init__(self):
        """
        Initialize an empty priority queue.
        """
        self.queue = []

    def insert(self, node):
        """
        Insert a PixelNode into the priority queue.

        :param node: The PixelNode to be inserted
        """
        self.queue.append(node)
        node_i = len(self.queue)-1
        node.heap_index = node_i
        self.heapify_up(node_i)
    
    def heapify_up(self, i):
        father_i = PriorityQueue.father(i)
        while i > 0 and self.queue[i] < self.queue[father_i]:
            self.queue[father_i].heap_index , self.queue[i].heap_index = i, father_i
            self.queue[father_i], self.queue[i] = self.queue[i], self.queue[father_i]
            i = father_i
            father_i = PriorityQueue.father(i)
        
        return i 


    def extract_min(self):
        """
        Extract and return the PixelNode with the smallest distance.

        :return: The PixelNode with the smallest distance
        """
        if not self.queue:
            return 
        self.queue[0], self.queue[-1] = self.queue[-1], self.queue[0]
        self.queue[0].heap_index = 0
        _min = self.queue.pop()
        self.heapify_down(0)
        return _min
    
    def heapify_down(self, i):
        heap_len = len(self.queue)
        left_child = PriorityQueue.left_child(i)
        while left_child < heap_len:
            minimum = i
            if self.queue[i] > self.queue[left_child]:
                minimum = left_child
            right_child = PriorityQueue.right_child(i)
            if right_child < heap_len and self.queue[right_child] < self.queue[minimum]:
                minimum = right_child

            if minimum == i:
                break

            self.queue[i].heap_index, self.queue[minimum].heap_index = minimum , i
            self.queue[i], self.queue[minimum] = self.queue[minimum], self.queue[i]
            i = minimum
            left_child = PriorityQueue.left_child(i)
        
        return i 

    def decrease_key(self, node, new_distance):
        """
        Update the distance of a node and reheapify.

        :param node: The PixelNode whose distance is to be updated
        :param new_distance: The new distance value
        """
        node.distance = new_distance
        node_i = node.heap_index
        node_i = self.heapify_up(node_i)
        self.heapify_down(node_i)

    

"""
4. Dijkstra's Algorithm
Implement Dijkstra's algorithm to find the shortest path from the start pixel to the end pixel.
"""



def dijkstra(gray_image_matrix, start, end, BOUNDARY = 200):
    """
    Perform Dijkstra's algorithm to find the shortest path in a maze.

    :param image: 2D NumPy array representing the grayscale maze
    :param start: Tuple (x, y) of the start pixel's coordinates
    :param end: Tuple (x, y) of the end pixel's coordinates
    :return: List of (x, y) tuples representing the shortest path
    """
    
    pixel_nodes_matrix = [
                        [PixelNode(x,y,cell) for x, cell in enumerate(row)]
                        for y, row in enumerate(gray_image_matrix)
                          ]

    x,y = start
    start_node = pixel_nodes_matrix[y][x]
    start_node.distance = 0
    start_node.visited = True

    queue = PriorityQueue()

    for row in pixel_nodes_matrix:
        for cell in row:
            queue.insert(cell)

    visited = set()

    while queue.queue:
        minimum = queue.extract_min()
        minimum.visited = True
        x, y = minimum.x, minimum.y
        visited.add((x,y))
        if (x,y) == end:
            break 
        minimum_neighbors = [pixel_nodes_matrix[i][j]
                            for i in range(y-1, y+2)
                              for j in range(x-1, x+2)
                                if 0 <= i < len(pixel_nodes_matrix) and 0 <= j < len(pixel_nodes_matrix[i])
                                  and (x,y) != (i,j)
                                  and not pixel_nodes_matrix[i][j].visited
                                  and pixel_nodes_matrix[i][j].color > BOUNDARY]
         
        for neighbor in minimum_neighbors:
            new_distance = minimum.distance + minimum._distance(neighbor)
            if new_distance < neighbor.distance:
                neighbor.prev = minimum
                queue.decrease_key(neighbor, new_distance)

    if not queue.queue:
        return []
    
    res = []
    x, y = end
    while (x,y) != start:
        res.append((x,y))
        prev = pixel_nodes_matrix[y][x].prev
        x, y = prev.x, prev.y

    res.append(start)
    print (f'{len(visited)=} {len(res)=}')
    return res[::-1], visited


########### util ########


def find_start_and_end(rgb_array):
    height, width, _ = rgb_array.shape
    start_point = None
    end_point = None

    white_color = (255, 255, 255)

    dominance_factor = 2.5

    for y in range(height):
        for x in range(width):
            r, g, b = rgb_array[y, x]

            if g > 150 and g > r * dominance_factor and g > b * dominance_factor and not start_point:
                start_point = find_nearby_white(rgb_array, x, y, white_color)

            elif r > 150 and r > g * dominance_factor and r > b * dominance_factor and not end_point:
                end_point = find_nearby_white(rgb_array, x, y, white_color)

            if start_point and end_point:
                return start_point, end_point

    if not start_point or not end_point:
        raise ValueError("Could not find both start and end points.")

    return start_point, end_point


def find_nearby_white(rgb_array, x, y, white_color):
    height, width, _ = rgb_array.shape
    directions = [(-20, 0), (20, 0), (0, -20), (0, 20)]  

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height:
            if tuple(rgb_array[ny, nx]) == white_color:
                return nx, ny

    raise ValueError(f"No nearby white pixel found for arrow at ({x}, {y})")


def mark_start_end_with_unique_color(rgb_array, start, end):
    UNIQUE_COLOR = (200, 255, 255)
    RADIUS = 100
    
    height, width, _ = rgb_array.shape
    
    prev_start = []
    prev_end = []

    def draw_circle(center, prev_colors):
        cx, cy = center
        for y in range(cy - RADIUS, cy + RADIUS + 1):
            for x in range(cx - RADIUS, cx + RADIUS + 1):
                if 0 <= x < width and 0 <= y < height:
                    if (x - cx) ** 2 + (y - cy) ** 2 <= RADIUS ** 2:
                        prev_colors.append((x, y, tuple(rgb_array[y, x].tolist()))) 
                        rgb_array[y, x] = UNIQUE_COLOR

    draw_circle(start, prev_start)
    draw_circle(end, prev_end)

    return prev_start, prev_end

def remove_unique_color_and_restore(rgb_array, prev_start, prev_end):
    for x, y, color in prev_end:
        rgb_array[y, x] = color
    for x, y, color in prev_start:
        rgb_array[y, x] = color


def flood_fill_from_corners(image):
    GRAY_COLOR = (50, 50, 50)
    UNIQUE_COLOR = (200, 255, 255)
    height, width, _ = image.shape
    
    def is_white(pixel):
        return np.all(pixel > 240)
    
    from collections import deque
    
    def flood_fill(start_x, start_y):
        queue = deque([(start_x, start_y)])
        
        while queue:
            x, y = queue.popleft()
            if not (0 <= x < width and 0 <= y < height):
                continue
                
            pixel = image[y, x]
            if not is_white(pixel):
                continue
            if tuple(image[y, x]) == GRAY_COLOR or tuple(image[y, x]) == UNIQUE_COLOR:
                continue
                
            image[y, x] = GRAY_COLOR
            
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                queue.append((x + dx, y + dy))
    
    corners = [(0, 0), (0, height-1), (width-1, 0), (width-1, height-1)]
    for corner_x, corner_y in corners:
        if is_white(image[corner_y, corner_x]):
            flood_fill(corner_x, corner_y)
            
    return image


##### run ######


def main(num):
    print(f"running...({num})")
    path = f'/Users/moshmish/Documents/test.mefatchim/תרגיל 4/mazes/maze{num}.jpg'
    img = Image.open(path, 'r')
    img = img.convert('RGB')
    rgb_array = np.array(img)

    start, end = find_start_and_end(rgb_array)
    prev_start, prev_end = mark_start_end_with_unique_color(rgb_array, start, end)

    flood_fill_from_corners(rgb_array)
    remove_unique_color_and_restore(rgb_array, prev_start, prev_end)
    
    rgb_list = rgb_array.tolist()
    gray_image_matrix = convert_to_grayscale(rgb_list)
    
    shortest_path, visited_cells = dijkstra(gray_image_matrix, start, end)

    img_output = Image.new('RGB', img.size)  
    img_output.paste(img)


    #צביעת פיקסלים שביקרתי בהם
    draw = ImageDraw.Draw(img_output)
    for x, y in visited_cells:
        if 0 <= x < img_output.width and 0 <= y < img_output.height:  
            draw.point((x, y), fill=(255, 255, 0))

    # (צביעת הנתיב הקצר ביותר באדום (בעובי 3 פיקסלים
    for x, y in shortest_path:
        for dx in range(-1, 2):
            width_x = x + dx
            if 0 <= width_x < img_output.width and 0 <= y < img_output.height:  
                draw.point((width_x, y), fill=(255, 0, 0))
    

    
    
    solution_path = f'/Users/moshmish/Documents/test.mefatchim/תרגיל 4/mazes/solution_maze_{num}.jpg'
    img_output.save(solution_path)
    print(f"Solution saved to {solution_path}")

    return


if __name__ == '__main__':
    for num in range(1, 2):
        main(num)


# Notes:
# - Ensure all data structures are used efficiently to minimize runtime.
# - Use helper functions where necessary to keep the code modular.
# - Test your implementation with various mazes and start/end points to ensure correctness.
#
# ---
# Submission:
# Submit your completed Python file with all the required functions and classes implemented. Include comments explaining your code where necessary.

