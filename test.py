import src.config as cfg
import src.utils.ine.ine_reduce as redu
import numpy as np
import cv2

proc1 = redu.INEReduce();

imgTest = cfg.PATH_BASE + "test_4.jpg"


image = cv2.imread(imgTest)


                    


proc1.reduce(imgTest, cfg.PATH_BASE + "salida.png")