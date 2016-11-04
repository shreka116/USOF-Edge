--
--  Copyright (c) 2016, Computer Vision Lab @ Seoul National University.
--  All rights reserved.
--
--


local M = {}

function M.parse(arg)
   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Torch-7 Unsupervised Optical Flow via CNN Training script')
--   cmd:text('See https://github.com/facebook/fb.resnet.torch/blob/master/TRAINING.md for examples')
   cmd:text()
   cmd:text('Options:')
    ------------ General options --------------------
--   cmd:option('-data',       '',         'Path to dataset')
   cmd:option('-dataset',      'FlyingChairs',  'Options: mpiSintel | KITTI2012 | FlyingChairs | UCF101')
   cmd:option('-manualSeed',         0,         'Manually set RNG seed')
   cmd:option('-nGPU',               1,         'Number of GPUs to use by default')
   cmd:option('-backend',         'cudnn',      'Options: cudnn | cunn')
   cmd:option('-cudnn',          'fastest',     'Options: fastest | default | deterministic')
   cmd:option('-genData',        '../dataset',  'Path to dataset')
   ------------- Data options ------------------------
   cmd:option('-nThreads',           2,         'number of data loading threads')
   ------------- Training options --------------------
   cmd:option('-nEpochs',            0,         'Number of total epochs to run')
   cmd:option('-epochNumber',        1,         'Manual epoch number (useful on restarts)')
   cmd:option('-batchSize',          8,        'mini-batch size (1 = pure stochastic)')
   cmd:option('-testOnly',         'false',     'Run on validation set only')
--    cmd:option('-tenCrop',         'false', 'Ten-crop testing')
   ------------- Checkpointing options ---------------
   cmd:option('-save',          'checkpoints',  'Directory in which to save checkpoints')
   cmd:option('-resume',           'none',      'Resume from the latest checkpoint in this directory')
   cmd:option('-retrain',            'none',         'Resume from the latest checkpoint in this directory')

   ---------- Optimization options ----------------------
   cmd:option('-learningRate',      1.6e-5,       'initial learning rate')
   cmd:option('-beta_1',            0.9,        'first parameter of Adam optimizer')
   cmd:option('-beta_2',           0.999,       'second parameter of Adam optimizer')
   ---------- Model options ----------------------------------
   cmd:option('-networkType',     'USOFnet',    'Options: USOFnet | Flownet | USCNNnet')
   cmd:option('-retrain',         'none',       'Path to model to retrain with')
   cmd:option('-optimState',      'none',       'Path to an optimState to reload from')
   ---------- Hyper parameters  ------------------------------
   cmd:option('-epsilon',           0.001,      'Usually we use 0.001')
   cmd:option('-photo_char',        0.25,       'Power of Chrabonnier Penalty, FlyingChairs = 0.25 , KITTI = 0.38')
   cmd:option('-smooth_char',       0.37,       'Power of Chrabonnier Penalty, FlyingChairs = 0.37 , KITTI = 0.21')
   cmd:option('-smooth_weight',       1,        'FlyingChairs = 1 , KITTI = 0.53')
   cmd:option('-edge_weight',       0.5,        'FlyingChairs = 0.5 , KITTI = 0.53')
   cmd:text()

   local opt = cmd:parse(arg or {})

   opt.testOnly = opt.testOnly ~= 'false'
--    opt.tenCrop = opt.tenCrop ~= 'false'
--    opt.shareGradInput = opt.shareGradInput ~= 'false'
--    opt.optnet = opt.optnet ~= 'false'
--    opt.resetClassifier = opt.resetClassifier ~= 'false'

   if not paths.dirp(opt.save) and not paths.mkdir(opt.save) then
      cmd:error('error: unable to create checkpoint directory: ' .. opt.save .. '\n')
   end

   if opt.dataset == 'mpiSintel' then
      -- Default nEpochs=90
      opt.nEpochs = opt.nEpochs == 0 and 90 or opt.nEpochs
   elseif opt.dataset == 'KITTI2012' then
      -- Default nEpochs=164
      opt.nEpochs = opt.nEpochs == 0 and 164 or opt.nEpochs
   elseif opt.dataset == 'FlyingChairs' then
       -- Default nEpochs=164
       opt.nEpochs = opt.nEpochs == 0 and 60 or opt.nEpochs
   elseif opt.dataset == 'UCF-101-100' then
       -- Default nEpochs=164
       opt.nEpochs = opt.nEpochs == 0 and 164 or opt.nEpochs
   else
      cmd:error('unknown dataset: ' .. opt.dataset)
   end

   return opt
end

return M
