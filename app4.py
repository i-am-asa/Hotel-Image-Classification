import json
from PIL import Image
from torchvision import models
import torch
import numpy as np
import streamlit as st
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from albumentations.pytorch import ToTensorV2
import albumentations as A
import pandas as pd
import matplotlib.pyplot as plt
import calendar


class ImgModel(nn.Module):
    def __init__(self, hidden_layer_size, num_classes):
        super().__init__()
        self.effnet = EfficientNet.from_pretrained('efficientnet-b4')
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.1)
        self.dense_layer1 = nn.Linear(1792, hidden_layer_size)
        self.norm1 = nn.BatchNorm1d(hidden_layer_size)
        self.out = nn.Linear(hidden_layer_size, num_classes)

    def forward(self, inputs):
        eff_out1 = self.effnet.extract_features(inputs['images'])
        eff_out1 = nn.Flatten()(self._avg_pooling(eff_out1))
        eff_out1 = self.drop(eff_out1)
        output = self.dense_layer1(eff_out1)
        output = self.norm1(output)
        output = self.out(output)
        output = nn.Softmax(dim=1)(output)
        return output


is_cpu = True
model_path = './models/val_loss_1.8715epoch_67.pth'
json_file = 'historical_data.json'

model = ImgModel(256, 15)
device = "cpu"
model.to(device)
params = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(params['model'])
model.eval()

Classes = ["Balcony", "Bar", "Bathroom", "Bedroom", "Bussiness Centre", "Dining room", "Exterior",
           "Gym", "Living room", "Lobby", "Patio", "Pool", "Restaurant", "Sauna", "Spa"]

class_counts = pd.DataFrame(index=Classes, columns=['Count'])
class_counts['Count'] = 0

class_to_ind = {}
ind_to_class = {}
for i, cl in enumerate(Classes):
    class_to_ind[cl] = i
    ind_to_class[i] = cl

# Load or create the historical data dictionary

try:
    with open(json_file, "r") as file:
        historical_data = json.load(file)
except (FileNotFoundError, json.JSONDecodeError):
    historical_data = {}

    
# set title of app
st.title("Hotel Image Classification App")
st.write("")

# enable users to upload images for the model to make predictions
file_up = st.file_uploader("Upload one or more images", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)


def plot_class_distribution():
    class_percentages = class_counts / class_counts.sum() * 100

    plt.figure(figsize=(8, 4))
    colors = ['#FFB74D', '#9CCC65', '#4DB6AC', '#64B5F6', '#7986CB', '#FF8A65', '#A1887F',
              '#F06292', '#9575CD', '#4FC3F7', '#BA68C8', '#FF8A80', '#FFD54F', '#81C784', '#4DD0E1']
    bar_width = 0.6

    # Plot the horizontal bar chart
    plt.barh(class_percentages.index, class_percentages['Count'], color=colors, height=bar_width)

    # Customize the appearance
    plt.xlabel('Percentage')
    plt.ylabel('Class')
    plt.title('Percentage of Each Class')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.xticks(rotation=0)
    plt.gca().invert_yaxis()  # Invert the y-axis to show classes from top to bottom

    # Add percentage labels on the bars
    for i, v in enumerate(class_percentages['Count']):
        plt.text(v + 1, i, f'{v:.2f}%', color='black', va='center')

    # Remove the spines
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    # Show the plot
    st.pyplot(plt.gcf())


def predict(image):
    """Return prediction with highest probability.
    Parameters
    ----------
    :param image: uploaded image
    :type image: jpg
    :rtype: list
    :return: Predicted class
    """
    # transform the input image through resizing, normalization
    transform = A.Compose([
        A.Resize(height=224, width=224),
        A.RandomCrop(height=200, width=200),
        A.HorizontalFlip(p=0.5),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.Blur(blur_limit=3, p=0.5),
        A.Normalize(),
        ToTensorV2(),
    ])

    # preprocess the image and make predictions
    img = Image.open(image).convert('RGB')
    img = transform(image=np.array(img))['image']
    img = img.unsqueeze(0)
    output = model({'images': img})
    probs = nn.Softmax(dim=1)(output)

    # get the predicted class label and probability
    max_prob, class_idx = torch.max(probs, dim=1)
    class_label = ind_to_class[class_idx.item()]

    return class_label


if file_up is not None:
    if st.button("Predict"):
        # Display the uploaded images and predictions
        for file in file_up:
            col1, col2, col3 = st.columns([3, 3, 3])
            with col1:
                st.write("")
            with col2:
                st.image(file, caption='Input Image', width=300)
            with col3:
                st.write("")

            class_label = predict(file)

            # Update the class count in the DataFrame
            class_counts.loc[class_label, 'Count'] += 1

            st.markdown("<h3 align=\"center\"><u>Prediction</u>:   " + class_label + "</h3>",
                        unsafe_allow_html=True)
            st.markdown(
                "<div style='text-align: center' <b>**********************************<b/></div>",
                unsafe_allow_html=True)

        # Save the updated class counts to the historical data dictionary
        for class_label, count in class_counts['Count'].items():
            if class_label in historical_data:
                historical_data[class_label].append(count)
            else:
                historical_data[class_label] = [count]

        # Save the updated historical data as JSON
        with open(json_file, "w") as file:
            json.dump(historical_data, file)

        # Plot the percentage graph after predictions
        st.header("Percentage of Each Class")
        plot_class_distribution()

# Perform trend analysis
# Perform trend analysis
st.header("Trend Analysis")

# Dropdown to select class
selected_class = st.selectbox("Select a class for trend analysis", Classes)

if selected_class:
    st.subheader(f"Trend Analysis - {selected_class}")

    if selected_class in historical_data:
        class_data = historical_data[selected_class]

        # Calculate the total counts
        total_counts = sum(class_data)

        if total_counts > 0:
            # Calculate the percentages
            percentages = [(count / total_counts) * 100 for count in class_data]

            # Calculate the increase or decrease in photos
            if len(class_data) > 1:
                diff = class_data[-1] - class_data[-2]
                if diff > 0:
                    change_text = f"Increase by {diff}"
                elif diff < 0:
                    change_text = f"Decrease by {abs(diff)}"
                else:
                    change_text = "No change"
            else:
                change_text = "No previous data"

            # Calculate the maximum and minimum values
            max_value = max(class_data)
            min_value = min(class_data)

            # Plot the trend graph
            plt.figure(figsize=(8, 5))
            plt.bar(range(1, len(class_data) + 1), class_data, width=0.4)
            plt.xlabel("Iteration (in months)")
            plt.ylabel("No. of images")
            plt.title(f"Trend Analysis for Class: {selected_class}\nChange: {change_text} since last iteration\nMax: {max_value}, Min: {min_value}")
            plt.grid(True)

            # Display the trend graph
            st.pyplot(plt.gcf())
        else:
            st.write("No historical data available for the selected class.")
    else:
        st.write("No historical data available for the selected class.")


import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

# Perform trend analysis on all classes combined
st.header("Trend Analysis - All Classes")

# Check if there is any historical data available
if len(historical_data) > 0:
    st.subheader("Trend Analysis - All Classes")

    # Calculate the total counts for each class
    total_counts = [sum(historical_data[cl]) for cl in Classes]

    # Calculate the percentages for each class
    percentages = [[(count / total_counts[i]) * 100 if total_counts[i] != 0 else 0 for count in historical_data[cl]] for i, cl in enumerate(Classes)]

    # Plot the trend graph
    plt.figure(figsize=(12, 8))

    # Iterate through each class and plot the trend
    for i, cl in enumerate(Classes):
        plt.plot(range(1, len(historical_data[cl]) + 1), percentages[i], label=cl)

    plt.xlabel("Iteration")
    plt.ylabel("Percentage")
    plt.title("Trend Analysis - All Classes")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)

    # Display the trend graph
    st.pyplot(plt.gcf())

    # Display important insights
    class_counts = [(cl, total_counts[i]) for i, cl in enumerate(Classes)]
    class_counts_sorted = sorted(class_counts, key=lambda x: x[1], reverse=True)

    st.subheader("Important Insights")
    # Calculate the total number of images uploaded
    total_images = sum(total_counts)

    # Calculate the increase or decrease in the image collection
    previous_count = sum(historical_data[Classes[0]][:-1]) if len(historical_data[Classes[0]]) > 1 else 0
    current_count = sum(historical_data[Classes[0]])
    change = current_count - previous_count

    if change > 0:
        change_text = f"Increase by {change} images"
        change_color = "green"
    elif change < 0:
        change_text = f"Decrease by {abs(change)} images"
        change_color = "red"
    else:
        change_text = "No change in image collection"
        change_color = "black"

    # Display the total number of images and the overall change in the image collection
    st.write("Total number of images uploaded:", f"<span style='font-weight:bold; font-size:16px;'>{total_images}</span>", unsafe_allow_html=True)
    st.write("Overall change in image collection:", f"<span style='color:{change_color}; font-weight:bold; font-size:16px;'>{change_text}</span>", unsafe_allow_html=True)
    st.write("Class with the most images:", f"<span style='font-weight:bold;'>{class_counts_sorted[0][0]}</span>", unsafe_allow_html=True)
    st.write("Count:", f"<span style='font-weight:bold;'>{class_counts_sorted[0][1]}</span>", unsafe_allow_html=True)
    st.write("Class with the least images:", f"<span style='font-weight:bold;'>{class_counts_sorted[-1][0]}</span>", unsafe_allow_html=True)
    st.write("Count:", f"<span style='font-weight:bold;'>{class_counts_sorted[-1][1]}</span>", unsafe_allow_html=True)

    growth_rates = []
    for cl in Classes:
        data = historical_data[cl]
        if data[0] != 0:  # Check if the initial count is not zero
            growth_rate = (data[-1] - data[0]) / data[0] * 100
        else:
            growth_rate = 0
        growth_rates.append((cl, growth_rate))
    growth_rates_sorted = sorted(growth_rates, key=lambda x: x[1], reverse=True)



else:
    st.write("No historical data available.")




# Calculate the growth percentages for each class
growth_percentages = [growth_rates_sorted[i][1] for i in range(len(growth_rates_sorted))]
class_names = [growth_rates_sorted[i][0] for i in range(len(growth_rates_sorted))]

# Filter classes that grew and classes that did not
growing_classes = [class_names[i] for i in range(len(class_names)) if growth_percentages[i] > 0]
non_growing_classes = [class_names[i] for i in range(len(class_names)) if growth_percentages[i] < 0]

# Calculate the growth percentages for growing classes and non-growing classes
growing_percentages = [growth_percentages[i] for i in range(len(class_names)) if growth_percentages[i] > 0]
non_growing_percentages = [abs(growth_percentages[i]) for i in range(len(class_names)) if growth_percentages[i] < 0]

# Check if there are growing classes
if len(growing_classes) > 0:
    # Plot the pie chart for growing classes
    plt.figure(figsize=(6, 6))
    plt.pie(growing_percentages, labels=growing_classes, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 8})
    plt.title("Classes with Growth")
    plt.axis('equal')
    st.pyplot(plt.gcf())
else:
    st.markdown("<p style='text-align: center; font-size: 20px'><b>No classes with growth</b></p>", unsafe_allow_html=True)

# Check if there are non-growing classes
if len(non_growing_classes) > 0:
    # Plot the pie chart for non-growing classes
    plt.figure(figsize=(6, 6))
    plt.pie(non_growing_percentages, labels=non_growing_classes, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 8})
    plt.title("Classes with No Growth")
    plt.axis('equal')
    st.pyplot(plt.gcf())
else:
    st.markdown("<p style='text-align: center; font-size: 20px'><b>No classes with no growth</b></p>", unsafe_allow_html=True)


#*********************** stacked area chart**********************

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

# Prepare the data for the stacked area chart
df_stacked = pd.DataFrame(historical_data, columns=Classes)

# Calculate the cumulative sum of each class
df_stacked = df_stacked.cumsum(axis=1)

# Set up the figure and axes
fig, ax = plt.subplots(figsize=(12, 8))

# Create the stacked area chart
ax.stackplot(range(1, len(df_stacked) + 1), df_stacked.T, labels=Classes)

# Set the labels and title
ax.set_xlabel("Iteration")
ax.set_ylabel("Count")
ax.set_title("Stacked Area Chart - Class Growth")

# Display the legend
ax.legend(loc="upper left")

# Display the chart
st.pyplot(fig)

#*********************** Comparison of Growth Rates**********************

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

# Prepare the data for the grouped bar chart
df_growth = pd.DataFrame(growth_rates, columns=["Class", "Growth Rate"])

# Set up the figure and axes
fig, ax = plt.subplots(figsize=(12, 8))

# Create the grouped bar chart
df_growth.plot(x="Class", y="Growth Rate", kind="bar", ax=ax)

# Set the labels and title
ax.set_xlabel("Class")
ax.set_ylabel("Growth Rate")
ax.set_title("Comparison of Growth Rates")

# Display the chart
st.pyplot(fig)

