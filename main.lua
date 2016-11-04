--
--  Copyright (c) 2016, Computer Vision Lab @ Seoul National University.
--  All rights reserved.
--
--
require 'gnuplot'
require 'torch'
require 'paths'
require 'optim'
require 'nn'

-- model 다시 짜야한다
local models        = require 'models/init'
local DataLoader    = require 'dataloader'
local opts          = require 'opts'
local Trainer    = require 'train'
local checkpoints= require 'checkpoints'

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)
cutorch.setDevice(1)

local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

-- Load previous checkpoint, if it exists
local checkpoint, optimState = checkpoints.latest(opt)

-- Create model
local model, edge_model, warp_image, warp_edge, smoothness_loss, criterion , edge_loss= models.setup(opt, checkpoint)

-- Data loading
local trainLoader, valLoader = DataLoader.create(opt)



-- The trainer handles the training loop and evaluation on validation set
local trainer = Trainer(model, edge_model, warp_image, warp_edge, smoothness_loss, criterion, edge_loss, opt, optimState)


--------TO DO -----------TO DO----------------
if opt.testOnly then
   local loss, avgEPE, errPixels = trainer:test(0, valLoader)
   print(string.format(' * Results loss: %1.4f  average EPE: %1.3f  %% of err. Pixels: %1.3f', loss, avgEPE, errPixels))
   return
end
---------------------------------------------

-- local trainLosses = {}
-- local testLosses = {}
--------TO DO -----------TO DO----------------
local avgEPE, errPixels = 0.0,0.0
local startEpoch = checkpoint and checkpoint.epoch + 1 or opt.epochNumber
local Losses     = checkpoint and torch.load('checkpoints/Losses_' .. startEpoch-1 .. '.t7') or {trainLosses = {}, testLosses = {}}
-- local trainLosses = checkpoint and torch.load('checkpoints/Loss_' .. startEpoch-1 .. 't7').trainLoss or {}
-- local testLosses = checkpoint and torch.load('checkpoints/Loss_' .. startEpoch-1 .. 't7').testLoss or {}

for epoch = startEpoch, opt.nEpochs do
   -- Train for a single epoch
   local trainLoss  = trainer:train(epoch, trainLoader)
   Losses.trainLosses[#Losses.trainLosses + 1] = trainLoss
   gnuplot.pngfigure('losses/trainLoss.png')
   gnuplot.plot({torch.range(1, #Losses.trainLosses), torch.Tensor(Losses.trainLosses), '-'})
   gnuplot.plotflush()
   -- Run model on validation set
--    local EPE_below, EPE_above, EPE_all, testLoss   = trainer:test(epoch, valLoader)
   local testLoss, averageEPE, erroneousPixels   = trainer:test(epoch, valLoader)
   Losses.testLosses[#Losses.testLosses + 1] = testLoss
   gnuplot.pngfigure('losses/testLoss.png')
   gnuplot.plot({torch.range(1, #Losses.testLosses), torch.Tensor(Losses.testLosses), '-'})
   gnuplot.plotflush()

--    local epeFile = 'EPE_' .. epoch .. '.t7'
--    torch.save(paths.concat(opt.save, epeFile), { EPE_below = EPE_below,
-- 						 EPE_above = EPE_above,
-- 						 EPE_all = EPE_all,
--                         --  testLoss = testLoss,
-- 						})
    avgEPE = averageEPE
    errPixel = erroneousPixels
   if (epoch%10 == 0) then
        checkpoints.save(epoch, model, trainer.optimState, opt)
        torch.save('checkpoints/Losses_' .. epoch .. '.t7', Losses)

   end

    print(string.format(' * Finished Epoch [%d/%d]  average EPE: %1.3f  %% of err. Pixels: %1.3f',epoch, opt.nEpochs, avgEPE,errPixel  ))

end

---------------------------------------------
