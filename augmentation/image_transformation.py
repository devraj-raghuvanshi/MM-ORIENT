import tensorflow as tf
import io
import os
import cv2
import pandas as pd

df = pd.read_csv('path/to/dataframe')

def augmentimage(image,name):
    image = cv2.imread(image)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.5)
    image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    # image = tf.image.random_crop(image, size=(image.shape[0]-50, image.shape[1]-50, 3))
    image = tf.image.random_hue(image, 0.2)
    image = tf.image.random_saturation(image, 5, 10)
    # image = tf.image.rot90(image, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    image = tf.keras.preprocessing.image.array_to_img(image.numpy())
    image.save("/path/to/output/directory/" +name)

def augmentall(input_folderimg):
  count=0
  names = []

  for img in list(set(df['image_name'])):
    df_temp = df.loc[df['image_name'] == img].reset_index(drop=True)
    for i,filename in enumerate(df_temp['image_name']):
      name="aug_" + str(i) + "_" + filename
      f = os.path.join(input_folderimg, filename)
      augmentimage(f,name)

      names.append(name)
      count += 1
      print(count, i, filename)

  return names

image_names = augmentall("/path/to/input/images/directory")

df.rename(columns={'image_name': 'parent_image'}, inplace=True)

filenames = []
for img in list(set(df['parent_image'])):
  df_temp = df.loc[df['parent_image'] == img].reset_index(drop=True)
  for i,filename in enumerate(df_temp['parent_image']):
    filenames.append(filename)

# Create a dictionary to map the "parent_image" values to their corresponding indices in the order_list
order_dict = {value: index for index, value in enumerate(filenames)}

# Add a temporary column for sorting based on the order_list
df["temp_sort_column"] = df["parent_image"].map(order_dict)

# Sort the DataFrame based on the temporary sorting column
sorted_df = df.sort_values("temp_sort_column")

# Drop the temporary sorting column
sorted_df = sorted_df.drop(columns=["temp_sort_column"])

sorted_df['image_name'] = image_names

sorted_df = sorted_df[['parent_image', 'image_name', 'original_text', 'rephrased_text', 'humour', 'sarcasm', 'offensive', 'motivational','overall_sentiment']].reset_index(drop=True)

sorted_df.to_csv('path/to/store/the/final/dataframe')