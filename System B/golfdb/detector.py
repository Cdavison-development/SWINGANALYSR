import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.options.display.width = None
pd.set_option("max_colwidth", None)
pd.options.display.max_rows = 999
import cv2
import pickle
import gzip
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn import svm, metrics, datasets
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from skimage.io import imread
from skimage.transform import resize
import warnings
warnings.filterwarnings("ignore")
import skimage
import os
from skimage.feature import hog
from skimage import exposure
rseed = 42
import imutils

def load_df(file_name):
    from scipy.io import loadmat
    import pandas as pd
    pd.options.display.width = None
    pd.set_option("max_colwidth", None)
    pd.options.display.max_rows = 999

    x = loadmat(file_name)
    l = list(x['golfDB'][0])
    d = dict()
    for idx, k in enumerate(l):
        d["{:3d}".format(idx)] = list(l[idx])
    df = pd.DataFrame(d).T
    df.columns = ["id", "youtube_id", "player", "sex", "club", "view", "slow", "events", "bbox", "split"]
    # 10 events = start_frame(SF), address(A), Toe-up(TU), Mid-backswing(MB), Top(T), Mid-downswing(MD), Impact(I),
    # Mid-follow-through(MFT), Finsh(F), end_frame(EF)
    # data format cleansing
    df['id'] = df['id'].apply(lambda x: x[0][0])
    df['youtube_id'] = df['youtube_id'].apply(lambda x: x[0])
    df['player'] = df['player'].apply(lambda x: x[0])
    df['sex'] = df['sex'].apply(lambda x: x[0])
    df['club'] = df['club'].apply(lambda x: x[0])
    df['view'] = df['view'].apply(lambda x: x[0])
    df['slow'] = df['slow'].apply(lambda x: x[0][0])
    df['events'] = df['events'].apply(lambda x: x[0])
    df['bbox'] = df['bbox'].apply(lambda x: x[0])
    df['split'] = df['split'].apply(lambda x: x[0][0])
    df = df.drop(columns=['split', 'youtube_id'])

    df.index = df.index.astype(int)
    # df.to_pickle('golfDB.pkl')
    df.to_csv('golfDB.csv')

    print("Number of annotations: {:3d}".format(len(df.id)))
    return df

def draw_bbox(id, df):
    video = cv2.VideoCapture("data/videos_160/" + str(id) + ".mp4")
    
    iterations = 0
    event_num = 1
    events = df.events[id]
#     print(type(df.bbox[id]))
    x, y, w, h = df.bbox[id]
    x, y, w, h = int(x*160), int(y*160), int(w*160), int(h*160) #make proportional to image 160 by 160
    label = ['Address', 'Toe-up', 'Mid-Backswing', 'Top', 'Mid-Downswing', 'Impact', 'Mid-Follow-Through', 'Finish']
    
    while True:
        ret, frame = video.read()

        if not ret:
            break
        if iterations == events[event_num] and event_num < 9:
            cv2.imwrite("Swing_events/" + label[event_num - 1] + "/" + str(id) + ".jpg", frame)
            event_num += 1
        iterations += 1
    video.release()
    df = load_df('../input/videos-160/golfDB.mat')
    print(df.head(16))
    for index in df.index:
        i = 0
        events = df.events[index]
        scaled_events = []
    for event in events:
        if i == 0:
            scaled_events.append(0)
        else:
            scaled_events.append(event - events[0])
        i += 1
    df.events[index] = scaled_events

    print(df.head(15))
    import shutil
    if os.path.exists("./Swing_events"):
        shutil.rmtree("./Swing_events")
    
    # create class folders
    os.makedirs('./Swing_events/Address')
    os.makedirs('./Swing_events/Toe-up')
    os.makedirs('./Swing_events/Mid-Backswing')
    os.makedirs('./Swing_events/Top')
    os.makedirs('./Swing_events/Mid-Downswing')
    os.makedirs('./Swing_events/Impact')
    os.makedirs('./Swing_events/Mid-Follow-Through')
    os.makedirs('./Swing_events/Finish')
    
    df = pd.read_pickle("./GolfDB.pkl")
    print(df.head(16))
    i = 0
    rows = []
    while i < 1400:
        draw_bbox(df.id[i], df)
        if (i % 100 == 0):
            print(i)
            i += 1

def load_image_files(container_path, dimension=(30, 30)):
    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]

    descr = "Your own dataset"
    images = []
    flat_data = []
    target = []
    for i, direc in enumerate(folders):
        for file in direc.iterdir():
            img = skimage.io.imread(file)
            img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
            flat_data.append(img_resized.flatten())
            images.append(img_resized)
            target.append(i)
    flat_data = np.array(flat_data)
    target = np.array(target)
    images = np.array(images)
    
    np.save('flat_dat.npy', flat_data)
    np.save('target.npy', target)
    np.save('target_names', categories)
    np.save('images.npy', images)
    np.save('descr.npy', descr)

    # return in the exact same format as the built-in datasets
    return Bunch(data=flat_data,
                 target=target,
                 target_names=categories,
                 images=images,
                 DESCR=descr)
    swing_image_dataset = load_image_files("./Swing_events/")