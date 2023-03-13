import os
import numpy as np
pq_coef = {}
pq_coef[4]={
    "dp":[[[1, -2, 0, 0, 0], 
          [0, 2, -6, 0, 0], 
          [0, -3, 9, -12, 0], 
          [0, 0, -4, 8,  0], 
          [0, 0, 0, -5, 15], 
          [0, 0, 0, 6, -12]],
         1
        ],#<---this 1 indicted we need divide q 1 time
    "dq":[
        [[1, 0, 0, 0, 0], 
         [-1, 0, 0, 0, 0], 
         [0, 0, -3, 0, 0], 
         [0, 0, 3, -8, 0], 
         [0, 0, -1, 4, 0], 
         [0, 0, 0, -2, 9], 
         [0, 0, 0, 2, -6]],
        2],#<---this 1 indicted we need divide q 2 time
    "de":[[[-1, 5, 0, 0, 0], 
          [1, -2, 0, 0, 0], 
          [0, 1, -3, 0, 0], 
          [0, -1, 3, -4, 0], 
          [0, 0, -1, 2, 0], 
          [0, 0, 0, -1, 3], 
          [0, 0, 0, 1, -2]],
          1]}
pq_coef[10]={
    "dp":[[[1, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 2, -6, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, -3, 9, -12, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0], [0, 0, -4, 12, -20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0], [0, 0, 0, -10, 30, -30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 
      6, -24, 36, -42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 7, -35, 
      70, -56, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 16, -56, 80, -72, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 27, -99, 135, -90, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0], [0, 0, 0, 0, 0, 0, -10, 60, -130, 140, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 
      0, 0, 0, 0, 0, 0, -11, 77, -176, 198, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 
      0, 0, 0, -24, 132, -252, 180, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 
      0, -39, 169, -286, 234, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -70, 
      266, -350, 182, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, -120, 
      315, -420, 225, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, -160, 
      416, -448, 128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 34, -221, 
      476, -459, 153, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 54, -306, 
      630, -522, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 95, -399, 
      627, -437, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 120, -500, 
      740, -420, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -21, 210, -588, 
      735, -378, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -22, 242, -682,
       814, -286, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -46, 345, -782, 
      713, -161, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -72, 408, -816, 
      744, -192, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -100, 500, -900, 
      575, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -130, 546, -884, 598, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -189, 702, -891, 405, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28, -252, 672, -784, 308, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 29, -290, 754, -754, 174, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 60, -360, 720, -660, 210, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 62, -372, 744, -558, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 96, -416, 640, -384, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      99, -429, 693, -297, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      136, -442, 510, -170, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      140, -455, 455, -210, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -36, 
      216, -432, 324, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -37, 
      185, -333, 259, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -38, 
      190, -304, 152, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -39, 
      156, -234, 195, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -40, 
      160, -200, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -41, 
      123, -123, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -42, 
      126, -168, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -43, 86, 
      0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -44, 132], [0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 45, -90]],
             1
            ],#<---this 1 indicted we need divide q 1 time
    "dq":[
            [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0], [0, 0, 3, -8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 
      0, -1, 6, -15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, -4, 
      18, -24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 2, -12, 24, -35, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 3, -20, 50, -48, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0], [0, 0, 0, 0, 0, 8, -35, 60, -63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 
      0, 0, 0, 0, 15, -66, 105, -80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 
      0, -5, 36, -91, 112, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, -6, 
      49, -128, 162, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, -14, 88, -189,
       150, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, -24, 117, -220, 198, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -45, 190, -275, 156, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 9, -80, 231, -336, 195, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 10, -110, 312, -364, 112, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 22, -156, 364, -378, 135, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 36, -221, 490, -435, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      65, -294, 495, -368, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      84, -375, 592, -357, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -14, 
      150, -448, 595, -324, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -15, 
      176, -527, 666, -247, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -32, 
      255, -612, 589, -140, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -51, 
      306, -646, 620, -168, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -72, 
      380, -720, 483, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -95, 
      420, -714, 506, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -140, 
      546, -726, 345, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, -189, 
      528, -644, 264, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21, -220, 
      598, -624, 150, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 44, -276, 
      576, -550, 182, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 46, -288, 
      600, -468, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 72, -325, 
      520, -324, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 75, -338, 
      567, -252, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 104, -351, 
      420, -145, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 108, -364, 
      377, -180, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -27, 168, -348, 270,
       0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -28, 145, -270, 217, 0, 0, 
      0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -29, 150, -248, 128, 0, 0, 0, 
      0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -30, 124, -192, 165, 0, 0, 0], [0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, -31, 128, -165, 0, 0, 0], [0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, -32, 99, -102, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, -33, 102, -140, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -34, 
      70, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -35, 108], [0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 35, -72]],
            2],#<---this 1 indicted we need divide q 2 time
    "de":[[[-1, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, -2, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0], [0, -1, 3, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 
      0, -1, 3, -5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, -2, 6, -6, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, -4, 6, -7, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0], [0, 0, 0, 0, 1, -5, 10, -8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0,
       0, 0, 0, 2, -7, 10, -9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 
      3, -11, 15, -10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, -1, 6, -13, 14,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, -1, 7, -16, 18, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0], [0, 0, 0, 0, 0, 0, 0, 0, -2, 11, -21, 15, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 
      0, 0, 0, 0, 0, 0, -3, 13, -22, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, -5, 19, -25, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -8, 
      21, -28, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -10, 26, -28, 
      8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -13, 28, -27, 9, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, -17, 35, -29, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 5, -21, 33, -23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 6, -25, 37, -21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 
      10, -28, 35, -18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 11, -31, 
      37, -13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 15, -34, 31, -7, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 17, -34, 31, -8, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, -4, 20, -36, 23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, -5, 21, -34, 23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, -7, 26, -33, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -9, 
      24, -28, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -10, 26, -26, 6,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -12, 24, -22, 7, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 2, -12, 24, -18, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 3, -13, 20, -12, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       3, -13, 21, -9, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, -13, 
      15, -5, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, -13, 13, -6, 0, 
      0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 6, -12, 9, 0, 0, 0, 0, 0, 
      0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, -1, 5, -9, 7, 0, 0, 0, 0, 0], [0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, -1, 5, -8, 4, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 
      4, -6, 5, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 4, -5, 0, 0, 
      0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 3, -3, 0, 0], [0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, -1, 3, -4, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, -1, 2, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 
      3], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -2]],
              1]}

def calculate_coef(e1,alpha0,alpha1,rank=4):
    """
    Input 
        e1     (B,)
        alpha0 (B,)
        alpha1 (B,)
    Ouput
        coef_of_error1 (B,)
        coef_of_error2 (B,)
        coef_of_error3 (B,)
    Basicly, we will assume you input a scalar, cause the e1,alpha0,alpha1 valid in `average` scenirio
    please use Mathematica to calculate the right coef table. We temperary record them here.
    Notice
        alpha1=(e3-e1)/e2;
        alpha0=(e2-e1)/e1
    TODO:
        this can be accelarate by Spare matrix multiply and batch multiply. 
        But since it only calculate one times. Can ignore? 
        Notice, we use CPU calculate
    """
    p = (1 - alpha1)
    q = (1/(1 - alpha0 +1e-5 ))
    e3= e1*(1 +  alpha1*(1 + alpha0))
    e2= e1*(1 + alpha0)
    e1= e1
    dp_matrix, dp_divide_times = pq_coef[rank]["dp"]
    dq_matrix, dq_divide_times = pq_coef[rank]["dq"]
    de_matrix, de_divide_times = pq_coef[rank]["de"]
    q_max_order = max(len(dp_matrix[0]),len(dq_matrix[0]),len(de_matrix[0]))
    p_max_order = max(len(dp_matrix)   ,len(dq_matrix)   ,len(de_matrix ))                              
    p_pows = np.array([np.power(p,n) for n in range(p_max_order)]) #(L)
    q_pows = np.array([np.power(q,n) for n in range(q_max_order)]) #(L)
    #print(p_pows.shape)
    dp  = np.einsum("i,ij,j->",p_pows[:len(dp_matrix)], np.array(dp_matrix), q_pows[:len(dp_matrix[0])])/np.power(q,dp_divide_times)*e1
    dq  = np.einsum("i,ij,j->",p_pows[:len(dq_matrix)], np.array(dq_matrix), q_pows[:len(dq_matrix[0])])/np.power(q,dq_divide_times)*e1
    de  = np.einsum("i,ij,j->",p_pows[:len(de_matrix)], np.array(de_matrix), q_pows[:len(de_matrix[0])])/np.power(q,de_divide_times)
    #print(dp,dq,de)
    de3 =  -(1/e2)*dp
    de2 = alpha1/e2*dp + 1/(e1*(1-alpha0)**2)*dq
    de  = 1/e2*dp - ((1/e1+alpha0/e1)/(1-alpha0)**2)*dq + de
    return de, de2, de3

def calculate_deltalog_coef(c1,c2,c3,e1,e2,e3):
    p1 = (c1+c2+c3)/(1+e1)
    p2 = (c2+c3)/(1+e2-e1)
    p3 = (c3)/(1+e3-e2)
    return p1,p2,p3

def normlized_coef_type1(coef1,coef2,coef3,*args):
    """
    Input 
        coef_of_error1 (B,)
        coef_of_error2 (B,)
        coef_of_error3 (B,)
    Ouput
        normed_coef_of_error1 (B,)
        normed_coef_of_error2 (B,)
        normed_coef_of_error3 (B,)
    """
    _sum = np.abs(coef1) + np.abs(coef2) + np.abs(coef3)
    return coef1/_sum,coef2/_sum,coef3/_sum
def normlized_coef_type2(c1,c2,c3,*args):
    """
    Input 
        c1,c2,c3
    Ouput
        c1,c2,c3
    erase value <0 
    min value 0.1
    """
    c1 = c1 if c1 >0 else 0
    c2 = c2 if c2 >0 else 0
    c3 = c3 if c3 >0 else 0
    norm   = np.sqrt(c1**2+c2**2+c3**2)
    c1,c2,c3 = c1/norm,c2/norm,c3/norm
    c1 = c1 if c1 >0.1 else 0.1
    c2 = c2 if c2 >0.1 else 0.1
    c3 = c3 if c3 >0.1 else 0.1
    return c1,c2,c3
def normlized_coef_type3(coef1,coef2,coef3,*args):
    """
    Input 
        coef_of_error1 a
        coef_of_error2 b
        coef_of_error3 c
    Ouput
        normed_coef_of_error1 a/sqrt(a**2+b**2+c**2) + 1
        normed_coef_of_error2 a/sqrt(a**2+b**2+c**2) + 1
        normed_coef_of_error3 a/sqrt(a**2+b**2+c**2) + 1
    """
    all_coef = np.sqrt(coef1**2 + coef2**2 + coef3**2)
    coef1 = coef1/all_coef + 1 
    coef2 = coef2/all_coef + 1 
    coef3 = coef3/all_coef + 1 
    return  coef1,coef2,coef3

def normlized_coef_type0(coef1,coef2,coef3,*args):
    all_coef = np.sqrt(coef1**2 + coef2**2 + coef3**2)
    coef1 = coef1/all_coef 
    coef2 = coef2/all_coef 
    coef3 = coef3/all_coef 
    return  coef1,coef2,coef3

def normlized_coef_type_bonded(c1,c2,c3,e1,e2,e3,delta=0.01):
    """
        get c1, c2, c3 for df = c1*de1 + c2*de2 + c3*de3
        the first thing is get the norm for variable
            v1 = ln(e1+1)
            v2 = ln(e2+1)
            v3 = ln(e3+1)
        ====> df = (e1+1)c1*dv1 + (e2+1)c2*dv2 + (e3+1)c3*dv3
        then we need apply a boundary compute 
        the boundary condition is 
            0< v2 - v1 < ln(2)
            0< v3 - v1 < ln(3)
            0< v1
        this can be accomplish by do a `reflect` vector 
            let z2 = (v2 - v1)/ln(2)
            let z3 = (v3 - v1)/ln(3)
            if z2 -> 0, this mean we need apply a gradient to increse it. 
                Notice in gradient update, we do x = x - lr*grad. Thus, the gradient direction should be -1
                thus, z2 -> 0 mean grad_of_z2 -> -10
            if z2 -> ln(2), grad_of_z2 -> 10. 

            can use Sinh(5*(2*z2-1))/Sinh(5) as the coef. if want to use much strong constrain replace 5 to 10
        This mean we need add two more constrain which lead
            c1 = (e1+1)c1 -> (e1+1)c1/N - Sinh(5*(2*z2-1))/Sinh(5) - Sinh(5*(2*z3-1))/Sinh(5)
            c2 = (e2+1)c2 -> (e2+1)c2/N + Sinh(5*(2*z2-1))/Sinh(5)
            c3 = (e3+1)c3 -> (e3+1)c3/N + Sinh(5*(2*z3-1))/Sinh(5)

    """
    # notice the error input must be mse e1 , but the c1,c2,c3 is calculated from e1 e2 e3
    c1 = (e1 + delta)*c1
    c2 = (e2 + delta)*c2
    c3 = (e3 + delta)*c3
    # do normlization 
    N = np.sqrt(c1**2 + c2**2 + c3**2)
    c1 = c1/N
    c2 = c2/N
    c3 = c3/N
    # apply boundary
    # #############
    v1 = np.log(e1) # <--- this is the ln loss we want to optimize
    v2 = np.log(e2) # <--- this is the ln loss we want to optimize
    v3 = np.log(e3) # <--- this is the ln loss we want to optimize
    #print(f"v1:{v1:.4f} v2:{v2:.4f} v3:{v3:.4f}") 
    z2 = (v2 - v1)/np.log(np.sqrt(2)) # <--- this is the boundary
    z3 = (v3 - v1)/np.log(np.sqrt(3)) # <--- this is the boundary
    #print(f"z2:{z2:.4f} z3:{z3:.4f}")
    z2 = (2*z2 - 1)
    z3 = (2*z3 - 1)
    alpha = 5
    cc2= np.sinh(alpha*z2)/np.sinh(alpha)
    cc3= np.sinh(alpha*z3)/np.sinh(alpha)
    #print(f"c1:{c1:.4f} c2:{c2:.4f} z2:{z2:.4f} z3:{z3:.4f} c3:{c3:.4f} cc2:{cc2:.4f} cc3:{cc3:.4f}")
    c1 = c1 - e1/(e1+delta)*cc2 - e1/(e1+delta)*cc3
    c2 = c2 + e2/(e2+delta)*cc2 
    c3 = c3 + e3/(e3+delta)*cc3
    #print(f"c1:{c1:.4f} c2:{c2:.4f} c3:{c3:.4f} cc2:{cc2:.4f} cc3:{cc3:.4f}")
    # apply normal constrain
    # we will add this offset untial the smallest one is np.sqrt(3)/3
    # c = np.array([c1,c2,c3])
    # minimal_index = np.array(c).argmin()
    # factor = (1 - c[minimal_index])/(np.sqrt(3)/3) - 1
    factor = 1
    c1+= factor*np.sqrt(3)/3
    c2+= factor*np.sqrt(3)/3
    c3+= factor*np.sqrt(3)/3
    N = np.sqrt(c1**2 + c2**2 + c3**2)
    c1 = c1/N
    c2 = c2/N
    c3 = c3/N
    return  c1,c2,c3