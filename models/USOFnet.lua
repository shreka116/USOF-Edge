--
--  Copyright (c) 2016, Computer Vision Lab @ Seoul National University.
--  All rights reserved.
--
--  Architecture borrowed from FlowNet:Simple
--
--  Fischer, Philipp, et al. "Flownet: Learning optical flow with convolutional networks."
--  arXiv preprint arXiv:1504.06852 (2015).
--

--require 'nn'
require 'nngraph'
require 'cunn'
require 'cudnn'
require 'math'
local nninit = require 'nninit'

local numChannels = 3
local Conv      = cudnn.SpatialConvolution
local deConv    = cudnn.SpatialFullConvolution
local ReLU      = cudnn.ReLU
local MaxPool   = cudnn.SpatialMaxPooling
local UpSample  = nn.SpatialUpSamplingNearest

local function createModel(opt)
    
    local model         = nn.Sequential()
    local model_par     = nn.ParallelTable()
    

    -- Contractive part of the network
    -- model for IMAGES
    local IM_concat_1, IM_concat_2, IM_concat_3, IM_concat_4, IM_concat_5   = nn.ConcatTable(), nn.ConcatTable(), nn.ConcatTable(), nn.ConcatTable(), nn.ConcatTable()
    local IM_seq_1, IM_seq_2, IM_seq_3, IM_seq_4, IM_seq_5, IM_seq_6        = nn.Sequential(), nn.Sequential(), nn.Sequential(), nn.Sequential(), nn.Sequential(), nn.Sequential()
    
    -- define the coarsest layer (9th conv.)
    IM_seq_6:add(Conv(512, 1024, 3, 3, 1, 1, 1, 1))
    IM_seq_6:add(MaxPool(2,2,2,2))
    IM_seq_6:add(ReLU(true))
    IM_concat_5:add(nn.Identity())
    IM_concat_5:add(IM_seq_6)

    -- 7th & 8th conv
    IM_seq_5:add(Conv(512, 512, 3, 3, 1, 1, 1, 1))
    IM_seq_5:add(MaxPool(2,2,2,2))
    IM_seq_5:add(ReLU(true))
    IM_seq_5:add(Conv(512, 512, 3, 3, 1, 1, 1, 1))
    IM_seq_5:add(ReLU(true))
    IM_seq_5:add(IM_concat_5)
    IM_concat_4:add(nn.Identity())
    IM_concat_4:add(IM_seq_5)
    
    -- 5th & 6th conv
    IM_seq_4:add(Conv(256, 512, 3, 3, 1, 1, 1, 1))
    IM_seq_4:add(MaxPool(2,2,2,2))
    IM_seq_4:add(ReLU(true))
    IM_seq_4:add(Conv(512, 512, 3, 3, 1, 1, 1, 1))
    IM_seq_4:add(ReLU(true))
    IM_seq_4:add(IM_concat_4)
    IM_concat_3:add(nn.Identity())
    IM_concat_3:add(IM_seq_4)

    -- 3rd & 4th conv
    IM_seq_3:add(Conv(128, 256, 5, 5, 1, 1, 2, 2))
    IM_seq_3:add(MaxPool(2,2,2,2))
    IM_seq_3:add(ReLU(true))
    IM_seq_3:add(Conv(256, 256, 3, 3, 1, 1, 1, 1))
    IM_seq_3:add(ReLU(true))
    IM_seq_3:add(IM_concat_3)
    IM_concat_2:add(nn.Identity())
    IM_concat_2:add(IM_seq_3)

    -- 2nd conv
    IM_seq_2:add(Conv(64, 128, 5, 5, 1, 1, 2, 2))
    IM_seq_2:add(MaxPool(2,2,2,2))
    IM_seq_2:add(ReLU(true))
    IM_seq_2:add(IM_concat_2)
    IM_concat_1:add(nn.Identity())
    IM_concat_1:add(IM_seq_2)

    -- 1st conv
    IM_seq_1:add(Conv(6, 64, 7, 7, 1, 1, 3 ,3))
    -- IM_seq_1:add(Conv(8, 64, 7, 7, 1, 1, 3 ,3))
    IM_seq_1:add(MaxPool(2,2,2,2))
    IM_seq_1:add(ReLU(true))
    IM_seq_1:add(IM_concat_1)
    IM_seq_1:add(nn.FlattenTable())

    -- model for EDGES
    -- local HED_model = torch.load('checkpoints/HED_pretrain.t7')

    model_par:add(IM_seq_1)
    model_par:add(nn.Identity())

    -- Expanding part of the network

    -- 1st & 2nd expansion
    local EX_seq_1, EX_seq_2, EX_seq_3                                               = nn.Sequential(), nn.Sequential(), nn.Sequential()
    local EX_concat_1_1, EX_concat_1_2, EX_concat_2_1, EX_concat_3_1, EX_concat_3_2  = nn.ConcatTable(), nn.ConcatTable(), nn.ConcatTable(), nn.ConcatTable(), nn.ConcatTable()
    local EX_concat_1_seq_1, EX_concat_1_seq_2, EX_concat_1_seq_3, EX_concat_1_seq_4, EX_concat_2_seq_1, EX_concat_2_seq_2, EX_concat_3_seq_1, EX_concat_3_seq_2 = nn.Sequential(), nn.Sequential(), nn.Sequential(), nn.Sequential(), nn.Sequential(), nn.Sequential(), nn.Sequential(), nn.Sequential()
    EX_concat_1_seq_1:add(nn.SelectTable(1))
    EX_concat_1_seq_1:add(nn.SelectTable(5))
    EX_concat_1_seq_2:add(nn.SelectTable(1))
    EX_concat_1_seq_2:add(nn.SelectTable(6))
    EX_concat_1_seq_2:add(deConv(1024, 512, 1, 1, 2, 2, 0, 0, 1, 1))
    EX_concat_1_seq_2:add(ReLU(true))
    EX_concat_1_seq_3:add(deConv(1024, 512, 1, 1, 2, 2, 0, 0, 1, 1))
    EX_concat_1_seq_3:add(ReLU(true))
    EX_concat_1_seq_4:add(UpSample(2))
    EX_concat_1_seq_4:add(Conv(1024, 2, 5, 5, 1, 1, 2, 2))
    EX_concat_1_1:add(EX_concat_1_seq_1)
    EX_concat_1_1:add(EX_concat_1_seq_2)
    EX_concat_1_2:add(EX_concat_1_seq_3)
    EX_concat_1_2:add(EX_concat_1_seq_4)
    EX_seq_1:add(EX_concat_1_1)
    EX_seq_1:add(nn.JoinTable(2))
    EX_seq_1:add(EX_concat_1_2)
    EX_seq_1:add(nn.JoinTable(2))
    
    EX_concat_2_seq_1:add(nn.SelectTable(1))
    EX_concat_2_seq_1:add(nn.SelectTable(4))
    EX_concat_2_seq_2:add(nn.SelectTable(2))
    EX_concat_2_seq_2:add(nn.SelectTable(5))
    EX_concat_2_1:add(EX_concat_2_seq_1)
    EX_concat_2_1:add(EX_concat_2_seq_2)
    EX_seq_2:add(EX_concat_2_1)
    EX_seq_2:add(nn.JoinTable(2))

    EX_concat_3_seq_1:add(deConv(1027, 256, 1, 1, 2, 2, 0, 0, 1, 1))
    EX_concat_3_seq_1:add(ReLU(true))
    EX_concat_3_seq_2:add(UpSample(2))
    EX_concat_3_seq_2:add(Conv(1027, 2, 5, 5, 1, 1, 2, 2))
    EX_concat_3_1:add(EX_seq_1)
    EX_concat_3_1:add(EX_seq_2)
    EX_concat_3_2:add(EX_concat_3_seq_1)
    EX_concat_3_2:add(EX_concat_3_seq_2)
    EX_seq_3:add(EX_concat_3_1)
    EX_seq_3:add(nn.JoinTable(2))
    EX_seq_3:add(EX_concat_3_2)
    EX_seq_3:add(nn.JoinTable(2))


    -- 3rd expansion
    local EX_seq_4, EX_seq_5 = nn.Sequential(), nn.Sequential()
    local EX_concat_4_1, EX_concat_5_1, EX_concat_5_2 = nn.ConcatTable(), nn.ConcatTable(), nn.ConcatTable()
    local EX_concat_4_seq_1, EX_concat_4_seq_2, EX_concat_5_seq_1, EX_concat_5_seq_2 = nn.Sequential(), nn.Sequential(), nn.Sequential(), nn.Sequential()
    EX_concat_4_seq_1:add(nn.SelectTable(1))
    EX_concat_4_seq_1:add(nn.SelectTable(3))
    EX_concat_4_seq_2:add(nn.SelectTable(2))
    EX_concat_4_seq_2:add(nn.SelectTable(4))
    EX_concat_5_seq_1:add(deConv(515, 128, 1, 1, 2, 2, 0, 0, 1, 1))
    EX_concat_5_seq_1:add(ReLU(true))
    EX_concat_5_seq_2:add(UpSample(2))
    EX_concat_5_seq_2:add(Conv(515, 2, 5, 5, 1, 1, 2, 2))
    EX_concat_4_1:add(EX_concat_4_seq_1)
    EX_concat_4_1:add(EX_concat_4_seq_2)
    EX_seq_4:add(EX_concat_4_1)
    EX_seq_4:add(nn.JoinTable(2))
    EX_concat_5_1:add(EX_seq_3)
    EX_concat_5_1:add(EX_seq_4)
    EX_concat_5_2:add(EX_concat_5_seq_1)
    EX_concat_5_2:add(EX_concat_5_seq_2)
    EX_seq_5:add(EX_concat_5_1)
    EX_seq_5:add(nn.JoinTable(2))
    EX_seq_5:add(EX_concat_5_2)
    EX_seq_5:add(nn.JoinTable(2))

    -- 4th expansion
    local EX_seq_6, EX_seq_7 = nn.Sequential(), nn.Sequential()
    local EX_concat_6_1, EX_concat_7_1, EX_concat_7_2 = nn.ConcatTable(), nn.ConcatTable(), nn.ConcatTable()
    local EX_concat_6_seq_1, EX_concat_6_seq_2, EX_concat_7_seq_1, EX_concat_7_seq_2 = nn.Sequential(), nn.Sequential(), nn.Sequential(), nn.Sequential()
    EX_concat_6_seq_1:add(nn.SelectTable(1))
    EX_concat_6_seq_1:add(nn.SelectTable(2))
    EX_concat_6_seq_2:add(nn.SelectTable(2))
    EX_concat_6_seq_2:add(nn.SelectTable(3))
    EX_concat_7_seq_1:add(deConv(259, 64, 1, 1, 2, 2, 0, 0, 1, 1))
    EX_concat_7_seq_1:add(ReLU(true))
    EX_concat_7_seq_2:add(UpSample(2))
    EX_concat_7_seq_2:add(Conv(259, 2, 5, 5, 1, 1, 2, 2))
    EX_concat_6_1:add(EX_concat_6_seq_1)
    EX_concat_6_1:add(EX_concat_6_seq_2)
    EX_seq_6:add(EX_concat_6_1)
    EX_seq_6:add(nn.JoinTable(2))
    EX_concat_7_1:add(EX_seq_5)
    EX_concat_7_1:add(EX_seq_6)
    EX_concat_7_2:add(EX_concat_7_seq_1)
    EX_concat_7_2:add(EX_concat_7_seq_2)
    EX_seq_7:add(EX_concat_7_1)
    EX_seq_7:add(nn.JoinTable(2))
    EX_seq_7:add(EX_concat_7_2)
    EX_seq_7:add(nn.JoinTable(2))

    -- 5th expansion
    local EX_seq_8, EX_seq_9 = nn.Sequential(), nn.Sequential()
    local EX_concat_8_1, EX_concat_9_1 = nn.ConcatTable(), nn.ConcatTable()
    local EX_concat_8_seq_1, EX_concat_8_seq_2 = nn.Sequential(), nn.Sequential()
    EX_concat_8_seq_1:add(nn.SelectTable(1))
    EX_concat_8_seq_1:add(nn.SelectTable(1))
    EX_concat_8_seq_2:add(nn.SelectTable(2))
    EX_concat_8_seq_2:add(nn.SelectTable(2))
    EX_concat_8_1:add(EX_concat_8_seq_1)
    EX_concat_8_1:add(EX_concat_8_seq_2)
    EX_seq_8:add(EX_concat_8_1)
    EX_seq_8:add(nn.JoinTable(2))
    EX_concat_9_1:add(EX_seq_7)
    EX_concat_9_1:add(EX_seq_8)
    EX_seq_9:add(EX_concat_9_1)
    EX_seq_9:add(nn.JoinTable(2))
    EX_seq_9:add(UpSample(2))

    -- local EX_seq_10 = nn.Sequential()
    -- EX_seq_10:add(nn.SelectTable(2))
    -- EX_seq_10:add(nn.SelectTable(1))

    
    -- -- Prediction layer
    -- local EX_seq_11 = nn.Sequential()
    -- local EX_concat_11_1 = nn.ConcatTable()
    -- EX_concat_11_1:add(EX_seq_9)
    -- EX_concat_11_1:add(EX_seq_10)
    -- EX_seq_11:add(EX_concat_11_1)
    -- EX_seq_11:add(nn.JoinTable(2))
    -- EX_seq_11:add(Conv(132, 2, 5, 5, 1, 1, 2, 2))

    EX_seq_9:add(Conv(131, 2, 5, 5, 1, 1, 2, 2))

    -- overall model
    model:add(model_par)
    -- model:add(EX_seq_11)
    model:add(EX_seq_9)


   local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
   end

   ConvInit('cudnn.SpatialConvolution')
   ConvInit('nn.SpatialConvolution')



    return model
end

return createModel