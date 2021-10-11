import os


def readname():
    # filePath = './div2k'
    # filePath = './pristine_images'
    # filePath = './BSD432_color'
    filePath = './flower_2000'
    name = os.listdir(filePath)
    return name, filePath


if __name__ == "__main__":
    name, filePath = readname()
    print(name)
    txt = open("train.txt", 'w')
    for i in name:
        image_dir = os.path.join(filePath + "/", str(i))
        txt.write(image_dir + "\n")
