import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import interpolate

def _update(i, imgs, cropped_imgs, points):
    global scat1
    if scat1 is not None:
        # clear scatterplots manually because they persist on the canvas
        scat1.remove()
        # clear the image drawn otherwise it will be very slow after several iterations
        ax1.cla()
        ax2.cla()
        ax3.cla()

    # Update the line and the axes (with a new xlabel). Return a tuple of
    # "artists" that have to be redrawn for this frame.
    ax1.set_title('Input cine - Frame {}'.format(i+1))
    ax2.set_title('Local ROI')
    ax3.set_title('Prediction')

    ax1.imshow(imgs[i], cmap='gray')
    ax2.imshow(cropped_imgs[i], cmap='gray')
    ax3.imshow(cropped_imgs[i], cmap='gray')
    
    # Turn off tick labels
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    ax3.set_yticklabels([])
    ax3.set_xticklabels([])

    dot_color1 = "red"
    point_size = 25
    scat1 = ax3.scatter(points[i][0], points[i][1], s=point_size, color=dot_color1)
    
    return [ax1, ax2, ax3]


def print_animation(img_sequence, cropped_sequence, coords_sequence, img_index, save_animation, gif_filepath):
    # print('Preparing animation for case', img_index)
    time_steps = 20
    frame_interval = 250

    anim = FuncAnimation(fig, _update, frames=np.arange(0, time_steps), interval=frame_interval, fargs=(img_sequence, cropped_sequence, coords_sequence))
    if (save_animation):
        anim.save(gif_filepath, writer='imagemagick', fps=5)
        print('Saved successfully in', gif_filepath)
    else:
        plt.show()
    plt.close()

def plot_strains(ax1, strains, strain_idx, strain_label):
    x = np.arange(1,21)
    y = strains[:,strain_idx] # get the corresponding strain
    
    x2 = np.linspace(x[0], x[-1], 100)
    y2 = interpolate.pchip_interpolate(x, y, x2)

    ax1.set_xlim([1, 20])
    ax1.axhline(0, color='black')
    ax1.set_xticks(np.arange(1, 20, step=2))

    ax1.set(xlabel='frame', ylabel='strain')
    ax1.plot(x, y, "o", color='black')
    ax1.plot(x2, y2, label=strain_label)
    ax1.legend()

def prepare_strain_chart(cc_strain, rr_strain):    
    ax4.set_title('Predicted strains')
    # Circumferential strain (0-6) endo to epi, 7 is global cc
    plot_strains(ax4, cc_strain, 7, r'Global $\epsilon_C$')
    plot_strains(ax4, cc_strain, 1, r'Subendo $\epsilon_C$')
    plot_strains(ax4, cc_strain, 3, r'Midwall $\epsilon_C$')
    plot_strains(ax4, cc_strain, 5, r'Subepi $\epsilon_C$')

    # Radial strain
    strain_idx = 0 # We only calculated 1 radial strain, the global one
    plot_strains(ax4, rr_strain, strain_idx, r'Global $\epsilon_R$')
    
def prepare_animation(img_sequence, local_images, landmark_sequence, cc_strains, rr_strains, save_to_gif, gif_filepath):
    global fig
    global scat1
    global ax1, ax2, ax3, ax4

    scat1 = None
    fig, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2,2, figsize=(8, 8))
    fig.set_tight_layout(True)

    # print('fig size: {0} DPI, size in inches {1}'.format(
    #     fig.get_dpi(), fig.get_size_inches()))

    prepare_strain_chart(cc_strains, rr_strains)
    print_animation(img_sequence, local_images, landmark_sequence, 0, save_to_gif, gif_filepath)