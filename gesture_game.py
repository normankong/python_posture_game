# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3.9.6 64-bit
#     name: python3
# ---

# # Gesture game using mediapipe library
# This game make use of the mediapipe as the posture detection.
#

# # To install dependencies
# - pip install mediapipe  
# - pip install numpy  
# - pip install cv2  
# - pip install pygame
# - pip install face_recognition
# - pip install trackeback

# ## Declare import and setup game Configuration

# +
from pathlib import Path
import traceback
import cv2
import mediapipe as mp
import numpy as np
import random 
import time
import pygame
import face_recognition
import traceback
import glob
import configparser

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
# +
# Game Config
config = configparser.ConfigParser(inline_comment_prefixes="#")
config.read('setting.ini')
FACES_FOLDER = config['DEFAULT']['FACES_FOLDER']
MUSIC_FOLDER = config['DEFAULT']['MUSIC_FOLDER']
ICONS_FOLDER = config['DEFAULT']['ICONS_FOLDER']
RADIUS = int(config['DEFAULT']['RADIUS'])
DIFFICULTIY = int(config['DEFAULT']['DIFFICULTIY'])
GAME_TIME = int(config['DEFAULT']['GAME_TIME'])
SPEED_INCREMENT = int(config['DEFAULT']['SPEED_INCREMENT'])
SPEED_MAXIMUM = int(config['DEFAULT']['SPEED_MAXIMUM'])
SPEED_INITIAL = int(config['DEFAULT']['SPEED_INITIAL'])
FRAME_SKIP_RATE = int(config['DEFAULT']['FRAME_SKIP_RATE'])
IS_DEBUG = (config['DEFAULT']['IS_DEBUG'] == 'True')

# Enum
GAME_TYPE_HAND = 0
GAME_TYPE_MOUTH = 1

# # Initialize Variable
class GameState:

    def __init__(self):
        self.is_finish = True
        self.img_x = (random.randint(100, 800))
        self.img_y = 0
        self.score = 0
        self.start_time = 0
        self.speed = SPEED_INITIAL
        self.elpase_time = 0
        self.game_type = GAME_TYPE_HAND
        self.icons = self.initIconsConfig()
        self.known_persons = self.initFaceConfig()
        self.name = "Unknown"
        self.face_recon_count = FRAME_SKIP_RATE
        self.current_icon = None
        pygame.mixer.init()
        pygame.mixer.music.load(MUSIC_FOLDER + "/background0.mp3")
        pygame.mixer.music.play()
    
    def initFaceConfig(self) -> tuple:   
        files = glob.glob(FACES_FOLDER + "/*")
        name_list = []
        encoding_list = []
        for x in files:
            image_file = face_recognition.load_image_file(x)
            face_encodings = face_recognition.face_encodings(image_file)[0]
            name_list.append(Path(x).stem)
            encoding_list.append(face_encodings)
        
        return { "name" : name_list, "encoding" : encoding_list }

    def initIconsConfig(self) -> list:   
        files = glob.glob(ICONS_FOLDER + "/*.png")
        result = []
        for x in files:
            image_file = cv2.imread(x, cv2.IMREAD_UNCHANGED)
            result.append(image_file)
        
        return result

    def getStateInfo(self) -> str:
        print("x: {}\ny: {}\nscore: {}\nspeed: {}\ngame_type: {}\nis_finish: {}\nname: {}".format(self.img_x, self.img_y, self.score, self.speed, self.game_type, str(self.is_finish), self.name))
        return ""

    def isInProgress(self) -> bool:
        return not self.is_finish

    def isFinish(self) -> bool:
        return self.is_finish
    
    def progressGame(self):
        self.elpase_time = GAME_TIME - (time.time() - self.start_time)
        if (self.elpase_time < 0):
            self.is_finish = True
            pygame.mixer.music.load(MUSIC_FOLDER + "/background0.mp3")
            pygame.mixer.music.play()
            return

        # Move the Object
        self.img_y += self.speed
        
        # Handle Overflow
        if (self.img_y + self.current_icon.shape[1] > self.image.shape[0]):
            self.img_y = 0
            self.img_x = (random.randint(100, 800))

    def startGame(self, game_type) -> None:
        self.is_finish = False
        self.img_x = (random.randint(100, 800))
        self.img_y = 0
        self.score = 0
        self.start_time = time.time()
        self.speed = SPEED_INITIAL
        self.elpase_time = 0
        self.game_type = game_type
        self.current_icon = random.choice(self.icons)
        if (self.game_type == GAME_TYPE_HAND):
            song = "/background1.mp3"
        if (self.game_type == GAME_TYPE_MOUTH):
            song = "/background2.mp3"
        pygame.mixer.music.load(MUSIC_FOLDER + song)
        pygame.mixer.music.play()

    def increaseLevel(self) -> None:
        pygame.mixer.Sound(MUSIC_FOLDER + "/win.wav").play()
        self.img_x = (random.randint(100, 800))
        self.img_y = 0
        self.score += 1
        self.speed += SPEED_INCREMENT
        if (self.speed > SPEED_MAXIMUM):
            self.speed = SPEED_MAXIMUM

    def isInRegion(self, a, b):
        a = np.array(a) # Point
        b = np.array(b) # Area

        xInRange = b[0] < a[0] < b[1] 
        yInRange = b[2] < a[1] < b[3]

        result = xInRange & yInRange
        return result

    def isReadyToStart(self, landmarks, area) -> bool:

        # If nothing detect, return
        if (landmarks is None):
            return False
        
        check_point = []

        # Check Left Hand
        check_point.append([landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * image.shape[1],landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * image.shape[0]])
        check_point.append([landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x * image.shape[1],landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].y * image.shape[0]])
        check_point.append([landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value].x * image.shape[1],landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value].y * image.shape[0]])
        check_point.append([landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value].x * image.shape[1],landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value].y * image.shape[0]])

        # Get Right Hand
        check_point.append([landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * image.shape[1],landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * image.shape[0]])
        check_point.append([landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x * image.shape[1],landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y * image.shape[0]])
        check_point.append([landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value].x * image.shape[1],landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value].y * image.shape[0]])
        check_point.append([landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value].x * image.shape[1],landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value].y * image.shape[0]])
        
        for x in check_point:
            if (self.isInRegion(x, area)):
                return True
        return False

    # Check if Hit with given point
    def isHit(self, pose_landmarks) -> bool:
        # If nothing detect, return
        if (pose_landmarks is None):
            return False

        landmarks = pose_landmarks.landmark

        check_point = []
        if (self.game_type == GAME_TYPE_MOUTH): 
            check_point.append([landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].x * image.shape[1],landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].y * image.shape[0]])
            check_point.append([landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].x * image.shape[1],landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].y * image.shape[0]])
            check_point.append([(check_point[0][0] + check_point[1][0])/2, (check_point[0][1] + check_point[1][1])/2])

        if (self.game_type == GAME_TYPE_HAND): 
            check_point.append([landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * image.shape[1],landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * image.shape[0]])
            check_point.append([landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x * image.shape[1],landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].y * image.shape[0]])
            check_point.append([landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value].x * image.shape[1],landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value].y * image.shape[0]])
            check_point.append([landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value].x * image.shape[1],landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value].y * image.shape[0]])

            # Get Right Hand
            check_point.append([landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * image.shape[1],landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * image.shape[0]])
            check_point.append([landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x * image.shape[1],landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y * image.shape[0]])
            check_point.append([landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value].x * image.shape[1],landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value].y * image.shape[0]])
            check_point.append([landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value].x * image.shape[1],landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value].y * image.shape[0]])
        
        # Determine whether in Range
        tmp_assistance = (100 - DIFFICULTIY)
        tmp_area = [self.img_x, self.img_x + RADIUS + tmp_assistance, self.img_y, self.img_y + RADIUS + tmp_assistance]  
        for x in check_point:
            if (self.isInRegion(x, tmp_area)):
                return True
        return False

def logoOverlay(image,logo,alpha=1.0,x=0, y=0, scale=1.0):
    (h, w) = image.shape[:2]
    image = np.dstack([image, np.ones((h, w), dtype="uint8") * 255])
    overlay = cv2.resize(logo, None,fx=scale,fy=scale)
    (wH, wW) = overlay.shape[:2]
    output = image.copy()
    # blend the two images together using transparent overlays
    try:
        if x<0 : x = w+x
        if y<0 : y = h+y
        if x+wW > w: wW = w-x  
        if y+wH > h: wH = h-y
        print(x,y,wW,wH)
        overlay=cv2.addWeighted(output[y:y+wH, x:x+wW],alpha,overlay[:wH,:wW],1.0,0)
        output[y:y+wH, x:x+wW ] = overlay
    except Exception as e:
        print("Error: Logo position is overshooting image!")
        print(e)
    output= output[:,:,:3]
    return output


# ## Main Logic
state = GameState()
# +
# Start Camera
cap = cv2.VideoCapture(0)

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        state.image = image
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        
        try:
            # Is Face available
            state.is_face_available = (results.pose_landmarks is not None)

            # Detect username if Face available
            if (state.is_face_available):
                
                # Increase Frame Count
                state.face_recon_count += 1
                if (state.face_recon_count == FRAME_SKIP_RATE + 1):
                    state.face_recon_count = 0

                    # Resize frame of video to 1/4 size for faster face recognition processing
                    small_image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)

                    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                    rgb_small_image = small_image[:, :, ::-1]

                    face_locations = face_recognition.face_locations(rgb_small_image)
                    face_encodings = face_recognition.face_encodings(rgb_small_image, face_locations)

                    known_face_encodings = state.known_persons["encoding"]
                    known_face_names = state.known_persons["name"] 

                    face_names = []
                    for face_encoding in face_encodings:
                        # See if the face is a match for the known face(s)
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                        state.name = "Unknown"

                        if True in matches:
                            first_match_index = matches.index(True)
                            state.name = known_face_names[first_match_index]

            # Process the game is not finish
            if (state.isInProgress()):
                if (state.isHit(results.pose_landmarks)):
                    state.increaseLevel()

                # Progress the Game
                state.progressGame()

                # Draw the Icon (Convert back to RGB temporarly)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image[ state.img_y:state.img_y + state.current_icon.shape[0] , state.img_x:state.img_x + state.current_icon.shape[1]] = state.current_icon
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    
            # Display Score
            cv2.rectangle(image, (0,0), (150,80), (245,117,16), -1)
            cv2.putText(image, state.name + '\'s score', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(state.score), (50,70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

            # Determine Score Board
            if (state.isFinish()):
                message = "Gameover"
            else:
                message = '{:.2f}'.format(state.elpase_time) + "s"
            cv2.rectangle(image, (image.shape[1] - 200,0), (image.shape[1], 73), (245,117,16), -1)
            cv2.putText(image, message, (image.shape[1] - 180,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

            # Display Debug
            if (IS_DEBUG):
                cv2.putText(image, state.getStateInfo(), (0,400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

            # Display the Game Start Option
            if (state.isFinish()) :

                 # Left screen, Right hand           
                cv2.rectangle(image, (0, image.shape[0]-100), (200, image.shape[0]), (102,0,255), -1)
                cv2.putText(image, "Hand", (50, image.shape[0]- 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                larea = (0, 200, image.shape[0] - 100, image.shape[0])

                # Right Screen, left hand
                cv2.rectangle(image, (image.shape[1] - 200, image.shape[0]-100), (image.shape[1], image.shape[0]), (3,213,123), -1)
                cv2.putText(image, "Mouth", (image.shape[1] - 150, image.shape[0]- 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                rarea = (image.shape[1] - 200, image.shape[1], image.shape[0] - 200, image.shape[0])

                if (not state.is_face_available) :
                    continue

                if (state.isReadyToStart(results.pose_landmarks.landmark, larea)):
                    state.startGame(GAME_TYPE_HAND)
                    print("Start Hand Game")
                    continue

                if (state.isReadyToStart(results.pose_landmarks.landmark, rarea)):
                    state.startGame(GAME_TYPE_MOUTH)
                    print("Start Mouth Game")
                    continue


        except Exception as e:
            print("Exception occured : ", e)
            traceback.print_exc()
            pass

        if (IS_DEBUG):
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Display the Frame
        cv2.imshow('Posture Game', image)
        
        if cv2.waitKey(5) & 0xFF == 27:
            break
    
cap.release()
# -

#
