import os

dir = 'results/CoSal2015'
dir = 'work_dirs/DINOV2-Proxy-32-SOD-Interaction-cos-global-sift-50-3-3-SAM-VIT_B/CoSal2015'
dir = 'data/CoCA/gt'
cnt = 0 

for root, dirs, files in os.walk(dir):
    for file in files:
        if 'png' in file: cnt = cnt + 1

print(cnt)

# CoCA img:1295 gt:1295 sod:1295 feat:1295
# CoSal2015 img:2015 gt:2015 sod:2015 feat:1295