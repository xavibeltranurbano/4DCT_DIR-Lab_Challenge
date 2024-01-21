# # -----------------------------------------------------------------------------
# # Preprocessing of the segmentation
# # Author: Frederik Hartmann, Yusuf Baran Tanrıverdi and Xavier Beltran Urbano
# # Date Created: 01-03-2023
# # -----------------------------------------------------------------------------


from matplotlib import pyplot as plt
import matplotlib as mpl
import plotly.graph_objects as go
from scipy.ndimage import zoom
import cv2
import numpy as np

class Ploting:
    @staticmethod
    def plot_3d(image,filename):
        image = zoom(1*(image), (0.2,0.2,0.2))
        # get range
        z, y, x = [np.arange(i) for i in image.shape]

        # meshgrid
        X,Y,Z = np.meshgrid(x,y,z, indexing='ij')

        fig = go.Figure(data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=np.transpose(image,(1,2,0)).flatten(),
            isomin=0.1,
            opacity=1, # needs to be small to see through all surfaces
            surface_count=1, # needs to be a large number for good volume rendering
            ))
        fig.write_html(filename)

    @staticmethod
    def scanSlice(scanSlice, figname=None):
        plt.imshow(scanSlice, cmap="gray")
        plt.axis("off")
        plt.savefig(figname)
        #plt.show()

    @staticmethod
    def contours(scanSlice, contours, filename):
        fig, ax = plt.subplots()
        ax.imshow(scanSlice, cmap="gray")
        for contour in contours:
            reshaped_contour = contour.squeeze()
            ax.plot(reshaped_contour[:, 1], reshaped_contour[:, 0], linewidth=2)
        #plt.show()
        plt.tight_layout()
        plt.savefig(filename)

    @staticmethod
    def comparison(scanSlice, GT, pred):
        # for coloring
        GT_inv = (np.logical_not(GT)).astype("uint8")
        pred_inv = (np.logical_not(GT)).astype("uint8")
        GT = GT.astype("uint8")
        pred = pred.astype("uint8")

        tp = cv2.bitwise_and(GT,pred)
        fp = cv2.bitwise_and(GT_inv,pred)
        fn = cv2.bitwise_and(GT,pred_inv)

        cmap1 = mpl.colors.ListedColormap(['none', 'green'])
        cmap2 = mpl.colors.ListedColormap(['none', 'red'])
        cmap3 = mpl.colors.ListedColormap(['none', 'blue'])


        plt.imshow(scanSlice, cmap="gray")
        plt.imshow(tp,  cmap=cmap1, alpha=0.5*(tp>0), interpolation="none")
        plt.imshow(fp,  cmap=cmap2, alpha=0.5*(fp>0), interpolation="none")
        plt.imshow(fn,  cmap=cmap3, alpha=0.5*(fn>0), interpolation="none")

        patch1 = plt.Rectangle((0, 0), 1, 1, color=cmap1(1.0))
        patch2 = plt.Rectangle((0, 0), 1, 1, color=cmap2(1.0))
        patch3 = plt.Rectangle((0, 0), 1, 1, color=cmap3(1.0))

        # add the legend to the plot
        plt.legend([patch1, patch2, patch3], ["True Positive", "False Positive","False Negative"], loc='upper right')
        plt.axis("off")
        plt.show()
    
    @staticmethod
    def trackingComparision(currentMask, prevMask, name):
        # for coloring
        tp = cv2.bitwise_and(currentMask,prevMask)
        xor = cv2.bitwise_xor(currentMask, prevMask)

        cmap1 = mpl.colors.ListedColormap(['none', 'green'])
        cmap2 = mpl.colors.ListedColormap(['none', 'red'])

        plt.imshow(tp,  cmap=cmap1, alpha=0.5*(tp>0), interpolation="none")
        plt.imshow(xor,  cmap=cmap2, alpha=0.5*(xor>0), interpolation="none")

        patch1 = plt.Rectangle((0, 0), 1, 1, color=cmap1(1.0))
        patch2 = plt.Rectangle((0, 0), 1, 1, color=cmap2(1.0))

        # add the legend to the plot
        plt.legend([patch1, patch2], ["Overlapping", "Difference"], loc='upper right')
        plt.axis("off")
        plt.show()

    @staticmethod
    def plot_data_with_filter(original_data, filtered_data, filter_name,):
        """
        Plots the original data and the filtered data.

        :param original_data: The original dataset.
        :param filtered_data: The dataset after applying the filter.
        :param filter_name: Name of the filter used.
        :param window_size: The window size used for the filter.
        :param poly_order: The polynomial order used for the filter.
        :param filename: The filename to save the plot.
        """
        plt.figure(figsize=(12, 6))
        indices = np.linspace(1, len(original_data), len(original_data))
        plt.plot(indices, original_data, marker='o', alpha=0.5, label='Original Data')
        plt.plot(indices, filtered_data, color='orange', label=f'Filtered)')
        plt.title(f'Data vs. Index (Original and {filter_name} Filtered)')
        plt.xlabel('Index')
        plt.ylabel('Data')
        plt.legend()
        plt.grid(True)

        # Save the plot
        plt.savefig(filter_name + ".png")
        plt.close()