--
--  Copyright (c) 2016, Computer Vision Lab @ Seoul National University.
--  All rights reserved.
--
--  Downloads data from web and save them as .t7 extensions
--
require 'utils'

local image = require 'image'
local paths = require 'paths'
local ffi = require 'ffi'
local t = require 'datasets/transforms'

local M = {}
local UCF_101_100 = torch.class('USOF.UCF_101_100', M)

function UCF_101_100:__init(imageInfo, opt, split)
   self.imageInfo = imageInfo[split]
   self.opt = opt
   self.split = split
--    self.dir = paths.concat(opt.data, split)
--    assert(paths.dirp(self.dir), 'directory does not exist: ' .. self.dir)
end

function UCF_101_100:get(i)
   
    -- local image_ref = self.imageInfo.image[{ {i},{1},{},{} }]
    -- local image_tar = self.imageInfo.image[{ {i},{2},{},{} }]

--    local path_ref = ffi.string(self.imageInfo.imagePath[{ {i},{1},{} }]:data())
--    local path_tar = ffi.string(self.imageInfo.imagePath[{ {i},{2},{} }]:data())

--    local image_ref = self:_loadImage(path_ref)
--    local image_tar = self:_loadImage(path_tar)

    -- print(path_ref)
    -- print(path_tar)

   local image = self.imageInfo.image[{ {i},{},{},{} }]:reshape(2,self.imageInfo.image[{ {i},{},{},{} }]:size(3),self.imageInfo.image[{ {i},{},{},{} }]:size(4))
--    local image = torch.cat(image_ref, image_tar, 1)

--    local path_flow = ffi.string(self.imageInfo.imageFlow[{ {i},{} }]:data())
   
--    local image_flow = readFlowFile(path_flow)
   local image_flow = torch.zeros(image:size())
    -- print(path_flow)
  

   return {
    input     = image,
    gt_flow   = image_flow,
   }
end

-- function UCF_101_100:_loadImage(path)
--    local ok, input = pcall(function()
--       return image.load(path, 3, 'float')
--    end)

--    -- Sometimes image.load fails because the file extension does not match the
--    -- image format. In that case, use image.decompress on a ByteTensor.
--    if not ok then
--       local f = io.open(path, 'r')
--       assert(f, 'Error reading: ' .. tostring(path))
--       local data = f:read('*a')
--       f:close()

--       local b = torch.ByteTensor(string.len(data))
--       ffi.copy(b:data(), data, b:size(1))

--       input = image.decompress(b, 3, 'float')
--    end

--    return input
-- end

function UCF_101_100:size()
   return self.imageInfo.image:size(1)
end

-- Computed from random subset of ImageNet training images
-- local meanstd = {
--    mean = { 0.485, 0.456, 0.406 },
--    std = { 0.229, 0.224, 0.225 },
-- }
-- local pca = {
--    eigval = torch.Tensor{ 0.2175, 0.0188, 0.0045 },
--    eigvec = torch.Tensor{
--       { -0.5675,  0.7192,  0.4009 },
--       { -0.5808, -0.0045, -0.8140 },
--       { -0.5836, -0.6948,  0.4203 },
--    },
-- }

function UCF_101_100:preprocess()
   if self.split == 'train' then
        return t.SelectTransform{
            -- t.AdditiveGausNoise(0, 0.04),
            -- t.Contrast(-0.8, 0.4),
            -- t.MultiplicativeColorChange(0.5, 2),
            -- t.AdditiveBrightness(0.2),
            -- t.GammaChanges(0.7, 1.5),
            t.Translations(0.2),
            t.HorizontalFlip(1),
            t.Rotation(17),
        --  t.Lighting(0.1, pca.eigval, pca.eigvec),
        --  t.ColorNormalize(meanstd),
        --  t.HorizontalFlip(0.5),
        }
    --   return t.Compose{
        --  t.ColorJitter({
        --     brightness = 0.4,
        --     contrast = 0.4,
        --     saturation = 0.4,
        --  }),
        --  t.Lighting(0.1, pca.eigval, pca.eigvec),
        --  t.ColorNormalize(meanstd),
        --  t.HorizontalFlip(0.5),
    --   }
   elseif self.split == 'val' then
      local Crop = self.opt.tenCrop and t.TenCrop or t.CenterCrop
        return t.SelectTransform{
            t.ColorNormalize(meanstd),
        }
    --   return t.Compose{
    --     --  t.Scale(256),
    --      t.ColorNormalize(meanstd),
    --     --  Crop(224),
    --   }
   else
      error('invalid split: ' .. self.split)
   end
end

return M.UCF_101_100
