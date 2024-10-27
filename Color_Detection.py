import cv2
import numpy as np
import pandas as pd
import argparse

def parse_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required=True, help="Path to the image")
    return vars(ap.parse_args())

def load_and_resize_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Could not load image")
    
    height, width = img.shape[:2]
    max_height, max_width = 800, 1200
    
    if height > max_height or width > max_width:
        scale = min(max_height/height, max_width/width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height))
    
    return img

def draw_text_with_outline(img, text, position, font_color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    outline_color = (0, 0, 0)  # Black outline
    outline_thickness = 3

    # Draw the outline
    cv2.putText(img, text, position, font, font_scale, outline_color, outline_thickness)
    # Draw the text
    cv2.putText(img, text, position, font, font_scale, font_color, font_thickness)

def get_color_name(R, G, B, csv_data):
    # Using a weighted Euclidean distance for better color matching
    minimum = float('inf')
    cname = "Not Found"
    
    # Weights for RGB components (human eye is more sensitive to green)
    weights = (0.3, 0.59, 0.11)
    
    for i in range(len(csv_data)):
        db = int(csv_data.loc[i, "B"]) - B
        dg = int(csv_data.loc[i, "G"]) - G
        dr = int(csv_data.loc[i, "R"]) - R
        
        # Weighted Euclidean distance
        d = np.sqrt(
            weights[0] * (dr ** 2) + 
            weights[1] * (dg ** 2) + 
            weights[2] * (db ** 2)
        )
        
        if d < minimum:
            minimum = d
            cname = csv_data.loc[i, "color_name"]
    
    return cname

def draw_function(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        global clicked, xpos, ypos
        clicked = True
        xpos = x
        ypos = y



def main():
    # Initialize global variables
    global clicked, xpos, ypos
    clicked = False
    xpos = ypos = 0
    
    # Load data and image
    args = parse_arguments()
    img = load_and_resize_image(args['image'])
    
    # Read color database
    index = ["color", "color_name", "hex", "R", "G", "B"]
    csv_data = pd.read_csv('colors.csv', names=index, header=None)
    
    # Create window and set mouse callback
    cv2.namedWindow('Color Detection')
    cv2.setMouseCallback('Color Detection', draw_function)
    
    while True:
        cv2.imshow('Color Detection', img)
        
        if clicked:
            # Create a small region around the clicked point for averaging
            y, x = ypos, xpos
            h, w = img.shape[:2]
            
            # Define a 5x5 region around the clicked point
            region_size = 2
            y1 = max(0, y - region_size)
            y2 = min(h, y + region_size + 1)
            x1 = max(0, x - region_size)
            x2 = min(w, x + region_size + 1)
            
            # Calculate average color in the region
            region = img[y1:y2, x1:x2]
            b, g, r = map(int, np.mean(region, axis=(0, 1)))
            
            # Calculate display positions
            rect_height = min(60, h // 10)
            rect_width = min(750, w - 40)
            rect_x = 20
            rect_y = 20
            
            # Create rectangle with detected color
            cv2.rectangle(img, (rect_x, rect_y), 
                        (rect_x + rect_width, rect_y + rect_height),
                        (b, g, r), -1)
            
            # Get color name
            color_name = get_color_name(r, g, b, csv_data)
            text = f'{color_name} R={r} G={g} B={b}'
            
            # Choose text color based on background brightness
            brightness = (r * 299 + g * 587 + b * 114) / 1000
            text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
            
            # Add text
            cv2.putText(img, text, (rect_x + 10, rect_y + 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                       text_color, 2, cv2.LINE_AA)
            
            clicked = False
        
        # Break loop on 'esc' key
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()