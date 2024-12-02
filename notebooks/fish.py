from glob import glob
import sys
import os
import random

from csbdeep.utils import normalize
from csbdeep.io import save_tiff_imagej_compatible

from stardist import random_label_cmap, _draw_polygons, export_imagej_rois
from stardist.models import StarDist2D

from dash import Dash, dcc, html, Input, Output
import porespy as ps

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from skimage import measure, morphology
from skimage.restoration import denoise_bilateral
from scipy.spatial import distance
from scipy import ndimage
import cv2

from tifffile import imread
import tifffile as tiff


class bc:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class LoadingBar:
    """
    Class to create a loading bar in the console.
    :param n_iters: number of iterations
    :param length: length of the loading bar
    """

    def __init__(self, n_iters, length=50):
        self.iter = 0
        self.n_iters = n_iters
        self.length = length

    def update(self):
        """
        Update the loading bar.
        """
        self.iter += 1
        progress = int((self.iter / self.n_iters) * self.length)
        sys.stdout.write(
            f'\r[{progress * "=" + (self.length - progress) * " "}] '
            f'{self.iter / self.n_iters:.2%}'
        )
        sys.stdout.flush()

    def end(self):
        """
        End the loading bar.
        """
        self.iter = 0
        sys.stdout.write('\n')
        sys.stdout.flush()


def read_tiff(path, axes='XYC'):
    """
    Read single TIFF file.
    :param path: Path to TIFF file.
    :param axes: Axes of the image. (Default: XYZ)
    :return: Image as numpy array.
    """
    if not path.endswith('.tiff') and not path.endswith('.tif'):
        return None, None

    img = np.array(imread(path))

    if axes == 'CYX':
        img = np.swapaxes(img, 0, 2)

    with tiff.TiffFile(path) as tif:
        first_page = tif.pages[0]

        resolution = first_page.tags.get('XResolution'), first_page.tags.get('YResolution')
        if resolution[0] is not None and resolution[1] is not None:
            x_res = resolution[0].value[0] / resolution[0].value[1] if resolution[0].value[0] != 0 else None
            y_res = resolution[1].value[0] / resolution[1].value[1] if resolution[1].value[0] != 0 else None
            z_res = 1.0
        else:
            x_res, y_res, z_res = None, None, None

    return img, {
        'x_res': x_res,
        'y_res': y_res,
        'z_res': z_res
    }


class FishAnalyzer:
    def __init__(
            self, folder,
            thr_size=300, thr_dist=15,
            model='2D_demo',
            verbose=0
    ):
        if folder.endswith('/'):
            folder = folder[:-1]

        assert os.path.exists(folder), 'Folder does not exist.'

        self.folder = folder
        self.paths = glob(f'{folder}/*.tif')
        self.thr_size = thr_size
        self.thr_dist = thr_dist
        self.verbose = verbose

        self.imgs = []
        self.metadata = []

        self.results = []
        self.results_df = []

        self.model_name = model
        self.model = None

    def load_images(self):
        assert len(self.paths) > 0, 'No images to load.'

        if self.verbose:
            print(f'{bc.OKBLUE}Loading images{bc.ENDC}...')
            bar = LoadingBar(len(self.paths))

        for path in self.paths:
            try:
                img, metadata = read_tiff(path)
                self.imgs.append(img)
                self.metadata.append(metadata)
                self.results_df.append(None)
                self.results.append(None)
            except Exception as e:
                print(f'{bc.WARNING}Error loading image{bc.ENDC}: {path}')
                print(f'{bc.BOLD}skipping{bc.ENDC}')

            if self.verbose:
                bar.update()

        if self.verbose:
            bar.end()
            print(f'{bc.OKGREEN}Images loaded.{bc.ENDC}')

    def load_model(self):
        assert self.model_name is not None, 'No model name provided.'

        self.model = StarDist2D.from_pretrained(self.model_name)

    def segment(self, img, name=None):
        assert self.model is not None, 'Model not loaded.'

        img_red = np.swapaxes(img[..., 0].astype(np.uint8), 0, 1)
        img_green = np.swapaxes(img[..., 1].astype(np.uint8), 0, 1)

        img_red_norm = normalize(img_red, 1, 99.8, axis=(0, 1)).astype(np.float32)
        img_green_norm = normalize(img_green, 1, 99.8, axis=(0, 1)).astype(np.float32)

        img_red_norm = ndimage.median_filter(img_red_norm, size=5)
        # img_red_norm = ndimage.median_filter(img_red_norm, size=5)
        img_green_norm = ndimage.median_filter(img_green_norm, size=5)
#         img_green_norm = ndimage.median_filter(img_green_norm, size=5)

        labels_red, _ = self.model.predict_instances(img_red_norm)
        labels_green, _ = self.model.predict_instances(img_green_norm)

        if name is not None:
            save_tiff_imagej_compatible(f'{self.folder}/results/{name}_red.tif', labels_red, axes='YXC')
            save_tiff_imagej_compatible(f'{self.folder}/results/{name}_green.tif', labels_green, axes='YXC')

        return labels_red, labels_green

    @staticmethod
    def filter_by_size(labels, thr=300):
        labeled = measure.label(labels)
        sizes = np.bincount(labeled.ravel())

        # Ignore background
        sizes[0] = 0

        large_particles = np.where(sizes > thr)[0]
        large_particles_mask = np.isin(labeled, large_particles)

        labels[large_particles_mask] = 0
        return labels

    @staticmethod
    def filter_by_intensity(img, seg, thr=25):
        intensities = []
        for label in np.unique(seg):
            if label == 0:
                continue

            mask = np.where(seg == label, img, 0).astype(np.uint8)
            coords = np.where(seg == label)
            margins = (
                np.min(coords, axis=1),
                np.max(coords, axis=1)
            )

            mask = mask[
                   int(margins[0][0]):int(margins[1][0]),
                   int(margins[0][1]):int(margins[1][1]),
                   ]
            intensities.append(np.median(mask))

        intensities = np.array(intensities)
        low_intensity_particles = np.where(intensities <= thr)
        low_intensity_particles_mask = np.isin(seg, low_intensity_particles)

        seg[low_intensity_particles_mask] = 0
        return seg

    @staticmethod
    def get_centroids(seg):
        props = ps.metrics.regionprops_3D(morphology.label(seg))
        centroids = [[round(i) for i in p.centroid] for p in props]
        centroids_labels = [seg[ce[0], ce[1]] for ce in centroids]

        res = []
        i = 0
        for p, label, centroid in zip(props, centroids_labels, centroids):
            res.append({
                'id': label,
                'centroid': centroid,
                'area': p.volume
            })

        return pd.DataFrame(res)

    @staticmethod
    def get_neighbours(data_source, data_target, tol=15):
        res = []
        for _, source_row in data_source.iterrows():
            source_id, source_centroid = source_row['id'], source_row['centroid']

            closest_dist = np.inf
            closest_id, closest_centroid, closest_area = None, None, None

            for _, target_row in data_target.iterrows():
                target_id, target_centroid = target_row['id'], target_row['centroid']

                dist = distance.euclidean(source_centroid, target_centroid)
                if dist < closest_dist and dist <= tol:
                    closest_dist = dist
                    closest_id = target_id
                    closest_centroid = target_centroid
                    closest_area = target_row['area']

            res.append({
                'red_id': source_id,
                'red_centroid': source_centroid,
                'red_area_pixels': source_row['area'],
                'green_id': closest_id,
                'green_centroid': closest_centroid,
                'green_area_pixels': closest_area,
                'distance_pixels': closest_dist
            })

        res = pd.DataFrame(res)
        res = res.dropna()

        res = res.loc[
            res.groupby('red_id')['distance_pixels'].idxmin()
        ].loc[
            res.groupby('green_id')['distance_pixels'].idxmin()
        ]

        res.reset_index(drop=True, inplace=True)
        return res

    def save(self, results, img, name, green_seg):
        plt.figure(figsize=(15, 7))

        plt.imshow(img, alpha=.6, cmap='gray')
        plt.axis('off')

        for _, pair in results.iterrows():
            source_centroid = pair['red_centroid']
            target_centroid = pair['green_centroid']
            dist = pair['distance_pixels']

            # color = [random.random() for _ in range(3)]
            color = 'red'

            plt.text(
                source_centroid[0] + 7, source_centroid[1] + 7,
                s=f'{dist:.1f}', color=color, fontsize='xx-small'
            )
            # plt.scatter(*source_centroid, color=color, alpha=1, marker='o', s=1, label='Red particle')
            # plt.scatter(*target_centroid, color=color, alpha=1, marker='x', s=1, label='Green particle')
            plt.plot(
                [source_centroid[0], target_centroid[0]],
                [source_centroid[1], target_centroid[1]],
                color=color, alpha=1, linewidth=1, label=f'Dist: {dist:.2f}'
            )

        if not os.path.exists(f'{self.folder}/results'):
            os.makedirs(f'{self.folder}/results')

        plt.savefig(f'{self.folder}/results/{name}.png', dpi=300)
        plt.close()

        if green_seg is None:
            # Tif image to save with green channel segmentation and red labels

            for i, label in enumerate(results['red_id']):
                # put text on the image with the id
                centroid = results.loc[i, 'red_centroid']
                cv2.putText(
                    green_seg, str(label), (int(centroid[0]), int(centroid[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
                )
            save_tiff_imagej_compatible(f'{self.folder}/results/{name}.tif', green_seg, axes='YXC')

        results.to_excel(f'{self.folder}/results/{name}.xlsx', index=False)

    def predict_all(self):
        assert len(self.imgs) > 0, 'No images to process.'
        assert self.model is not None, 'Model not loaded.'

        if self.verbose:
            print(f'{bc.OKBLUE}Predicting instances{bc.ENDC}...')
            bar = LoadingBar(len(self.imgs))

        for i, img in enumerate(self.imgs):
            try:
                img_name = self.paths[i].split('/')[-1].split('.')[0]
                labels_red, labels_green = self.segment(img, img_name)

                filtered_red = self.filter_by_size(labels_red, self.thr_size)
                filtered_green = self.filter_by_size(labels_green, self.thr_size)

                data_red = self.get_centroids(filtered_red)
                data_green = self.get_centroids(filtered_green)

                res = self.get_neighbours(data_red, data_green, self.thr_dist)

                img_name = self.paths[i].split('/')[-1].split('.')[0]
                self.save(res, img, img_name, labels_green)
                self.results_df[i] = res
            except Exception as e:
                print(f'\n{bc.FAIL}Error processing image{bc.ENDC}: {self.paths[i]}')
                print(f'{bc.BOLD}skipping{bc.ENDC}')
                import traceback
                traceback.print_exc()

            if self.verbose:
                bar.update()

        if self.verbose:
            bar.end()
            print(f'{bc.OKGREEN}Instances predicted and saved.{bc.ENDC}')

    def predict_single(self, img_path, filter=True):
        assert self.model is not None, 'Model not loaded.'

        if img_path not in self.paths:
            print(f'{bc.FAIL}Image not found{bc.ENDC}: {img_path}')
            return
        else:
            print(f'{bc.OKGREEN}Processing image{bc.ENDC}: {img_path}')

        img_name = img_path.split('/')[-1].split('.')[0]
        img_idx = self.paths.index(f'{self.folder}/{img_name}.tif')

        img = self.imgs[img_idx]
        labels_red, labels_green = self.segment(img, img_name)

        if filter:
            filtered_red = self.filter_by_size(labels_red, self.thr_size)
            filtered_green = self.filter_by_size(labels_green, self.thr_size)
        else:
            filtered_red, filtered_green = labels_red, labels_green

        data_red = self.get_centroids(filtered_red)
        data_green = self.get_centroids(filtered_green)

        res = self.get_neighbours(data_red, data_green, self.thr_dist)

        self.save(res, img, img_name, labels_green)
        try:
            self.results_df[img_idx] = res
        except IndexError:
            self.results_df.append(res)

        print(f'{bc.OKGREEN}Results saved.{bc.ENDC}')

    @staticmethod
    def plot_results_interactive(results, img):
        img_height, img_width, _ = img.shape

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Raw Image", "Predictions"),
            horizontal_spacing=0.05,
            shared_xaxes=True,
            shared_yaxes=True
        )

        fig.add_trace(go.Image(z=img, hoverinfo='skip'), row=1, col=1)
        fig.add_trace(go.Image(z=img, opacity=.5, hoverinfo='skip'), row=1, col=2)

        for idx, pair in results.iterrows():
            source_centroid = pair['red_centroid']  # [::-1]
            target_centroid = pair['green_centroid']  # [::-1]
            dist = pair['distance_pixels']

            fig.add_trace(go.Scatter(
                x=[source_centroid[0], target_centroid[0]],
                y=[source_centroid[1], target_centroid[1]],
                mode='lines+markers',
                line=dict(color='red', width=2.5),
                marker=dict(size=2),
                hovertext=(
                    f"Red({source_centroid[0]}, {source_centroid[1]})<br>"
                    f"Green({target_centroid[0]}, {target_centroid[1]})<br>"
                    f"Dist(px): {dist:.2f}"
                ),
                hoverinfo='text',
                customdata=[idx],
                showlegend=False
            ), row=1, col=2)

            midpoint = (
                (source_centroid[0] + target_centroid[0]) / 2,
                (source_centroid[1] + target_centroid[1]) / 2
            )
            offset = (15, 15)

            fig.add_trace(go.Scatter(
                x=[midpoint[0] + offset[0]],
                y=[midpoint[1] + offset[1]],
                mode='text',
                text=[f"{dist:.2f}"],
                textfont=dict(color='red', size=9),
                hoverinfo='skip',
                showlegend=False
            ), row=1, col=2)

        fig.update_layout(
            title="Interactive Particle Pairing",
            width=1200,
            height=600,
            xaxis=dict(
                scaleanchor="x2",
                range=[0, img_width],
                visible=False
            ),
            xaxis2=dict(
                scaleanchor="x",
                range=[0, img_width],
                visible=False,
            ),
            yaxis=dict(
                scaleanchor="y2",
                range=[0, img_height],
                autorange="reversed",
                visible=False,
            ),
            yaxis2=dict(
                scaleanchor="y",
                range=[0, img_height],
                autorange="reversed",
                visible=False,
            ),
            showlegend=False
        )

        return fig

    def analyze(self, img_path):

        if img_path not in self.paths:
            print(f'{bc.FAIL}Image not found{bc.ENDC}: {img_path}')
            return
        else:
            print(f'{bc.OKGREEN}Processing image{bc.ENDC}: {img_path}')

        img_name = img_path.split('/')[-1].split('.')[0]
        img_idx = self.paths.index(f'{self.folder}/{img_name}.tif')
        img = self.imgs[img_idx]
        res = pd.read_excel(f'{self.folder}/results/{img_name}.xlsx')

        print(f'{bc.OKGREEN}Results loaded.{bc.ENDC}')
        print(res.to_dict('records'))

        app = Dash(__name__)
        app.layout = html.Div([
            dcc.Graph(id='interactive-plot', config={'editable': False, 'displayModeBar': True}),
            html.Div(id='output', style={'marginTop': '20px'}),
            dcc.Store(id='results-store', data=res.to_dict('records')),
            dcc.Store(id='image-store', data=img.tolist())
        ])

        @app.callback(
            [
                Output('interactive-plot', 'figure'),
                Output('results-store', 'data')
            ],
            [
                Input('interactive-plot', 'clickData'),
                Input('results-store', 'data'),
                Input('image-store', 'data')
            ]
        )
        def update_plot(click_data, current_results, img_data):
            img = np.array(img_data)

            if current_results is None:
                return self.plot_results_interactive(pd.DataFrame(), img), []

            current_results = pd.DataFrame(current_results)

            if click_data is not None:
                print(click_data)
                clicked_idx = click_data['points'][0]['customdata']

                updated_results = current_results.drop(clicked_idx).reset_index(drop=True)
                self.save(updated_results, img, img_name, None)

                fig = self.plot_results_interactive(updated_results, img)
                return fig, updated_results.to_dict('records')

            print(img.shape)
            fig = self.plot_results_interactive(current_results, img)
            return fig, current_results.to_dict('records')

        app.run_server(debug=True)

