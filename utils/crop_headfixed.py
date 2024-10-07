from videotools import deviation_image, detect_blobs
import ipdb
import matplotlib.pyplot as plt

def main(file):
    images,devimage=deviation_image(file)
    plt.figure(figsize=(10,10))
    plt.imshow(devimage,cmap='gray')
    plt.show()

if __name__=='__main__':
    main(r'\\Kenneth-NAS\data\tmt_experiment_2024_working_file\Animals\C4635132_cohort-2_M2_control\day_0\24-07-09_day-0_C4635132_M2_side.avi')