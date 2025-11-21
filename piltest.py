import PIL
#print(PIL.__version__)

from PIL import Image

# 打开图像（支持 JPEG、PNG、GIF 等）
img = Image.open("/home/ddy/code/python/cvstudy/dog.jpeg")  

# 显示图像（系统默认图片查看器）
img.show()  

# 获取图像信息
print("格式:", img.format)  # JPEG/PNG
print("尺寸:", img.size)    # (宽, 高)
print("模式:", img.mode)   # RGB/L/CMYK