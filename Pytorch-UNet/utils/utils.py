import matplotlib.pyplot as plt


def plot_img_and_mask(img, img_procesat, mask):
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes + 2)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    ax[1].set_title('Img proc')
    ax[1].imshow(img_procesat)
    for i in range(classes):
        ax[i + 2].set_title(f'Mask (class {i + 2})')
        ax[i + 2].imshow(mask == i)
    plt.xticks([]), plt.yticks([])
    plt.show()
