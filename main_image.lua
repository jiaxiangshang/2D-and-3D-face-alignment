--[[
nvidia-docker run -it -v /data0/0_DATA:/data 1adrianb/facealignment-torch

docker exec -it cool_nightingale /bin/bash

# esrc
th main_image.lua -input /data/0_Face_3D/3_ESRC_OBJ/ -name_gl train_render.txt \
-type 3D -model /data/0_Face_3D/10_FAN/3D-FAN.t7  -output  /data/0_Face_3D/3_ESRC_OBJ \
-preffix_bbox _bbox -preffix_save _lm2d_v3

th main_image.lua -input /data/0_Face_3D/3_ESRC_OBJ/ -name_gl train_render.txt \
-type 3D -model /data/0_Face_3D/10_FAN/3D-FAN.t7  -output  /data/0_Face_3D/3_ESRC_OBJ \
-preffix_bbox _bbox_deca -preffix_save _lm2d_v3_deca

# GAN
th main_image.lua -input /data0/0_DATA/1_Face_2D/40_stylegan2_256_500000/ -name_gl train.txt -fmt_gl .png \
-type 3D -model /data/0_Face_3D/10_FAN/3D-FAN.t7  -output  /data0/0_DATA/1_Face_2D/40_stylegan2_256_500000 \
-preffix_bbox _bbox -preffix_save _lm2d_v3

th main_image.lua -input /data0/0_DATA/1_Face_2D/40_stylegan2_256_500000/ -name_gl train.txt -fmt_gl .png \
-type 3D -model /data/0_Face_3D/10_FAN/3D-FAN.t7  -output  /data0/0_DATA/1_Face_2D/40_stylegan2_256_500000 \
-preffix_bbox _bbox_deca -preffix_save _lm2d_v3_deca

# tencent test
th main_image.lua -input /data/1_Face_2D/20_tencent_video_GL/ -name_gl test.txt \
-type 3D -model /data/0_Face_3D/10_FAN/3D-FAN.t7  -output  /data0/0_DATA/1_Face_2D/20_tencent_video_GL \
-preffix_bbox _bbox -preffix_save _lm2d_v3

# tecent server
# celebA
CUDA_VISIBLE_DEVICES=1  th main_image.lua -input /apdcephfs/private_alexinwang/jxshang/data/0_3DFace_Train/2_mono/6_celebvoxel2/6_voxel_celeb2_GL/ -name_gl train.txt \
-type 3D -model /apdcephfs/private_alexinwang/jxshang/project/deeplearning_python/dl_model/3D-FAN.t7  -output /apdcephfs/private_alexinwang/jxshang/data/0_3DFace_Train/2_mono/6_celebvoxel2/6_voxel_celeb2_GL \
-preffix_bbox _bbox -preffix_save _lm2d_v3

th main_image.lua -input /apdcephfs/private_alexinwang/jxshang/data/0_3DFace_Train/2_mono/6_celebvoxel2/6_voxel_celeb2_GL/ -name_gl train.txt \
-type 3D -model /apdcephfs/private_alexinwang/jxshang/project/deeplearning_python/dl_model/3D-FAN.t7  -output /apdcephfs/private_alexinwang/jxshang/data/0_3DFace_Train/2_mono/6_celebvoxel2/6_voxel_celeb2_GL \
-preffix_bbox _bbox_deca -preffix_save _lm2d_v3_deca
--]]

require 'torch'
require 'nn'
require 'nngraph'
require 'paths'
require 'image'
require 'xlua'

local utils = require 'utils'
local opts = require 'opts'(arg)

-- Load optional libraries
require('cunn')
require('cudnn')

-- Load optional data-loading libraries
matio = xrequire('matio') -- matlab
-- npy4th = xrequire('npy4th') -- python numpy

local FaceDetector = require 'facedetection_dlib'

torch.setheaptracking(true)
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

string.split = function(s, p)
    local rt= {}
    string.gsub(s, '[^'..p..']+', function(w) table.insert(rt, w) end )
    return rt
end

function getGlobalList(data_path)
    print('getGlobalList...')
    local filesList = {}
    cnt = 1
    file = io.open(data_path,"r");
    for line in file:lines() do
        filesList[cnt] = line
        cnt = cnt + 1
    end
    file:close()
    print('Found '..#filesList..' images')
    return filesList
end

function getBBox(data_path)
    -- print('getBBox...')

    file = io.open(data_path,"r");
    for line in file:lines() do
        line_bbox = line
    end
    -- print(line_bbox)
    bbox = string.split(line_bbox, ',')
    file:close()
    --print('Found '..#bbox..' axis')
    return bbox
end

--CMD: th main_image.lua --input /data0/0_DATA/0_Face_3D/3_ESRC_OBJ/ --name_gl train_render.txt
print("Start landmark detection")
local data_top =  opts.input
local name_gl = opts.name_gl
local data_path = data_top..name_gl
local fileList = getGlobalList(data_path)
-- local fileList, requireDetectionCnt = utils.getFileList(opts)

local predictions = {}
local faceDetector = nil

-- if requireDetectionCnt > 0 then faceDetector = FaceDetector() end


--creat a cpu copy to clone model from it to gpu if limited gpu option is true
local cpuMemoryModel = torch.load(opts.model)
local model

if opts.limitedGpuMemory == 'true' and opts.device == 'gpu' then
  model = cpuMemoryModel:clone()
  
else 
  model = cpuMemoryModel
end

--if deveice spcified to be gpu 
if opts.device == 'gpu' then model = model:cuda() end


model:evaluate()


--if not limited Gpu memory and 3D-full prediction required then load the 3D depth prediction model to Gpu memory once
--other wise if limitedGpuMemory is true will load the depth model when needed and swap with th 2D prediction model to save some memory

local modelZ
local cpuMemoryModelZ

if opts.type == '3D-full' then 
    
  cpuMemoryModelZ = torch.load(opts.modelZ)
  
  if opts.device == 'gpu' and opts.limitedGpuMemory == 'false' then
      
    modelZ = cpuMemoryModelZ
    modelZ = modelZ:cuda()
    modelZ:evaluate()
		
  elseif opts.device == 'cpu'  then
  modelZ = cpuMemoryModelZ
  modelZ:evaluate()
  end
end

for i = 1, #fileList do

  collectgarbage()
  print("processing ",fileList[i])
    local list_gl = string.split(fileList[i], ' ')

    local name_subfolder = list_gl[1]
    local name_pure = list_gl[2]

    local path_image = data_top..name_subfolder..'/'..name_pure..opt.fmt_gl
    local path_bbox = data_top..name_subfolder..'/'..name_pure..opts.preffix_bbox..'.txt'
    local img = image.load(path_image)
    local detectedFace = getBBox(path_bbox)
    df_center, df_scale = utils.get_normalisation_bb(detectedFace)
    -- Convert grayscale to pseudo-rgb
    if img:size(1)==1 then
        print("\n-- Convert grayscale to pseudo-rgb")
        img = torch.repeatTensor(img,3,1,1)
    end
   
  -- Detect faces, if needed
  img = utils.crop(img, df_center, df_scale, 256):view(1,3,256,256)
    
  --cuda--
  if opts.device ~= 'cpu' then img = img:cuda() end
  
  local output = model:forward(img)[4]:clone()
  
  output:add(utils.flip(utils.shuffleLR(model:forward(utils.flip(img))[4])))
  
  local preds_hm, preds_img = utils.getPreds(output, df_center, df_scale)

  preds_hm = preds_hm:view(68,2):float()*4
  
  --there is no need to save the output in the gpu now 
  output = nil 
  collectgarbage()
  
  --if limited gpu memory is true now it's the time to swap between 2D and 3D prediction models 
  if opts.limitedGpuMemory=='true' and opts.type == '3D-full' and opts.device == 'gpu' then
    model = nil
    collectgarbage()
    modelZ = (cpuMemoryModelZ:clone()):cuda()
    modelZ:evaluate()    
    collectgarbage()
    
  end
  
  --proceed to 3D depth prediction (if the gpu limited memory is false then the 3D prediction model would be already loaded at the gpu memory at initialize steps and the 2D model stay at its place)
  -- depth prediction
  if opts.type == '3D-full' then
    out = torch.zeros(68, 256, 256)
    for i=1,68 do
      if preds_hm[i][1] > 0 then
          utils.drawGaussian(out[i], preds_hm[i], 2)
      end
    end
    out = out:view(1,68,256,256)
    local inputZ = torch.cat(img:float(), out, 2)

    if opts.device ~= 'cpu' then inputZ = inputZ:cuda() end
    
    local depth_pred = modelZ:forward(inputZ):float():view(68,1) 
    
    preds_hm = torch.cat(preds_hm, depth_pred, 2)
    preds_img = torch.cat(preds_img:view(68,2), depth_pred*(1/(256/(200*fileList[i].scale))),2)
    
  end
  
  --put back the 2D prediction model and pop the 3D model
  if opts.limitedGpuMemory=='true' and opts.type == '3D-full' and opts.device == 'gpu' then
    modelZ = nil
    collectgarbage()    
    model = (cpuMemoryModel:clone()):cuda()
    model:evaluate()
  end

  if opts.mode == 'demo' then
      
    if detectedFace ~= nil then
        -- Converting it to the predicted space (for plotting)
        detectedFace[{{3,4}}] = utils.transform(torch.Tensor({detectedFace[3],detectedFace[4]}), df_center, df_scale, 256)
        detectedFace[{{1,2}}] = utils.transform(torch.Tensor({detectedFace[1],detectedFace[2]}), df_center, df_scale, 256)

        detectedFace[3] = detectedFace[3]-detectedFace[1]
        detectedFace[4] = detectedFace[4]-detectedFace[2]
    end
    -- utils.plot(img, preds_hm, detectedFace)
  
  end

  if opts.save then
      -- local dest = opts.output..'/'..paths.basename(fileList[i].image, '.'..paths.extname(fileList[i].image))
      local dest = data_top..name_subfolder..'/'..name_pure
      if opts.outputFormat == 't7' then
        torch.save(dest..'.t7', preds_img)
      elseif opts.outputFormat == 'txt' then
        -- csv without header
        local out = torch.DiskFile(dest .. opts.preffix_save .. '.txt', 'w')
        out:writeString(tostring(68) .. '\n')
        for i=1,68 do
              --print(preds_img:size(1), preds_img:size(2))
              if preds_img:size(3)==3 then
                  out:writeString(tostring(preds_img[{1, i,1}]) .. ',' .. tostring(preds_img[{1, i,2}]) .. ',' .. tostring(preds_img[{1, i,3}]) .. '\n')
              else
                  out:writeString(tostring(preds_img[{1, i,1}]) .. ',' .. tostring(preds_img[{1, i,2}]) .. '\n')
              end
        end
        out:close()
      end
      xlua.progress(i, #fileList)
  end

  if opts.mode == 'eval' then
      predictions[i] = preds_img:clone() + 1.75
      xlua.progress(i,#fileList)
  end
	
  collectgarbage();

  ::continue::
      
end

if opts.mode == 'eval' then
    predictions = torch.cat(predictions,1)
    local dists = utils.calcDistance(predictions,fileList)
    utils.calculateMetrics(dists)
end
