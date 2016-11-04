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
local FlyingChairsDataset = torch.class('USOF.FlyingChairsDataset', M)

function FlyingChairsDataset:__init(imageInfo, opt, split)
   self.imageInfo = imageInfo[split]
   self.opt = opt
   self.split = split
--    self.dir = paths.concat(opt.data, split)
--    assert(paths.dirp(self.dir), 'directory does not exist: ' .. self.dir)
end

function FlyingChairsDataset:get(i)
   local path_ref = ffi.string(self.imageInfo.imagePath[{ {i},{1},{} }]:data())
   local path_tar = ffi.string(self.imageInfo.imagePath[{ {i},{2},{} }]:data())

   local image_ref = self:_loadImage(path_ref)
   local image_tar = self:_loadImage(path_tar)

    -- print(path_ref)
    -- print(path_tar)

   local image = torch.cat(image_ref, image_tar, 1)

   local path_flow = ffi.string(self.imageInfo.imageFlow[{ {i},{} }]:data())
   
   local image_flow = readFlowFile(path_flow)

    -- print(path_flow)
  

   return {
    input     = image,
    gt_flow   = image_flow,
   }
end

function FlyingChairsDataset:_loadImage(path)
   local ok, input = pcall(function()
      return image.load(path, 3, 'float')
   end)

   -- Sometimes image.load fails because the file extension does not match the
   -- image format. In that case, use image.decompress on a ByteTensor.
   if not ok then
      local f = io.open(path, 'r')
      assert(f, 'Error reading: ' .. tostring(path))
      local data = f:read('*a')
      f:close()

      local b = torch.ByteTensor(string.len(data))
      ffi.copy(b:data(), data, b:size(1))

      input = image.decompress(b, 3, 'float')
   end

   return input
end

function FlyingChairsDataset:size()
   return self.imageInfo.imagePath:size(1)
end

-- Computed from random subset of ImageNet training images
local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}
local pca = {
   eigval = torch.Tensor{ 0.2175, 0.0188, 0.0045 },
   eigvec = torch.Tensor{
      { -0.5675,  0.7192,  0.4009 },
      { -0.5808, -0.0045, -0.8140 },
      { -0.5836, -0.6948,  0.4203 },
   },
}

function FlyingChairsDataset:preprocess()
   if self.split == 'train' then
         return t.SelectTransform{
            -- photometric augmentation
            t.AdditiveGausNoise(0, 0.04),
            t.Contrast(-0.8, 0.4),
            t.MultiplicativeColorChange(0.5, 2),
            t.AdditiveBrightness(0.2),
            t.GammaChanges(0.7, 1.5),

            -- geometric augmentation
            t.Identity(),
            t.HorizontalFlip(0.5),
            t.Rotation_naive(17),
            t.Scales_naive(0.9, 2),
            t.Translations_naive(0.2),

            -- relative augmentation
            t.Rotation_naive(3),
            t.Scales_naive(1, 1.1),
            t.Translations_naive(0.05),

        }

   elseif self.split == 'val' then
          return t.SelectTransform{
              t.Identity(),
        }
   else
      error('invalid split: ' .. self.split)
   end
end

return M.FlyingChairsDataset
