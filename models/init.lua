--
--  Copyright (c) 2016, Computer Vision Lab @ Seoul National University.
--  All rights reserved.
--
--
--  Model creating code

require 'nn'
require 'cunn'
require 'cudnn'
require 'tvnorm-nn'
require 'stn'
require 'nngraph'

-- require 'BrightnessCriterion'
require '../AffineGridGeneratorUSOF'

local M = {}

function M.setup(opt, checkpoint)
    local model

    if checkpoint then
        local modelPath = paths.concat(opt.resume, checkpoint.modelFile)
        assert(paths.filep(modelPath), 'Saved model not found: ' .. modelPath)
        print('=> Resuming model from ' .. modelPath)
        model   = torch.load(modelPath):cuda()
    elseif opt.retrain ~= 'none' then
        assert(paths.filep(modelPath), 'Model not found: ' .. opt.retrain)
        print('=> Loading model from ' .. opt.retrain)
        model   = torch.load(opt.retrain):cuda()
    else
        print('=> Creating model from: models/' .. opt.networkType .. '.lua')
        model = require('models/' .. opt.networkType)(opt)
    end

   -- Set the CUDNN flags
   if opt.cudnn == 'fastest' then
      cudnn.fastest = true
      cudnn.benchmark = true
   elseif opt.cudnn == 'deterministic' then
      -- Use a deterministic convolution implementation
      model:apply(function(m)
         if m.setMode then m:setMode(1, 1, 1) end
      end)
   end    

   -- Wrap the model with DataParallelTable, if using more than one GPU
   if opt.nGPU > 1 then
      local gpus = torch.range(1, opt.nGPU):totable()
      local fastest, benchmark = cudnn.fastest, cudnn.benchmark

      local dpt = nn.DataParallelTable(1, true, true)
         :add(model, gpus)
         :threads(function()
            local cudnn = require 'cudnn'
            cudnn.fastest, cudnn.benchmark = fastest, benchmark
         end)
      dpt.gradInput = nil

      model = dpt:cuda()
   end

   local imgH   = 0
   local imgW   = 0

   if opt.dataset == "FlyingChairs" then
      imgH = 384
      imgW = 512
   elseif opt.dataset == "mpiSintel" then
      imgH = 436
      imgW = 1024
   elseif opt.dataset == "KITTI2012" then
      imgH = 384
      imgW = 512
   elseif opt.dataset == "UCF101" then
      imgH = 240
      imgW = 320
   end


   -- Define phtometric Loss
--    local photometric = nn.BrightnessCriterion():cuda()

   local epsilon             = opt.epsilon -- 0.001
   local char_power          = opt.photo_char
   local photometric         = nn.Sequential()
   local parallel_2          = nn.ParallelTable()
   local warp_image          = nn.Sequential()
   local warp_edge           = nn.Sequential()
   local parallel_1          = nn.ParallelTable()
   local trans               = nn.Sequential()
   trans:add(nn.Identity())
   trans:add(nn.Transpose({2,3},{3,4}))

   parallel_1:add(trans)
   parallel_1:add(nn.AffineGridGeneratorUSOF())

   warp_image:add(parallel_1)
   warp_image:add(nn.BilinearSamplerBHWD())
   warp_image:add(nn.Transpose({3,4},{2,3}))

   warp_edge:add(parallel_1)
   warp_edge:add(nn.BilinearSamplerBHWD())
   warp_edge:add(nn.Transpose({3,4},{2,3}))


   -- Define smoothness constraint
   local smoothness = nn.SpatialTVNormCriterion()

   local edgePenalty = nn.SmoothL1Criterion()

   local criterion  = nn.SmoothL1Criterion()

   local edge_model = torch.load('checkpoints/HED_pretrain_edge3.t7')

   model:cuda()
   edge_model:cuda()
   warp_image:cuda()
   warp_edge:cuda()
   smoothness:cuda()
   criterion:cuda()
   edgePenalty:cuda()

   cudnn.convert(warp_image, cudnn)
   cudnn.convert(warp_edge, cudnn)
   cudnn.convert(model, cudnn)
   cudnn.convert(edge_model, cudnn)

   edge_model:evaluate()

   return model, edge_model, warp_image, warp_edge, smoothness, criterion, edgePenalty

end

return M
