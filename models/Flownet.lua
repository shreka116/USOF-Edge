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


-- local f_depths  = {2*numChannels,64,128,256,256,512,512,512,512,1024,512,512,256,128,64,2} -- feature depths
-- local k_size    = { 7, 5,  5,  3,  3,  3,  3,  3,  3,   1,  1,  1,  1,  1, 1 } -- kernel size

local f_depths  = {2*numChannels,64,128,256,256,512,512,512,512,1024,1024,512,512,256,128,64,2} -- feature depths
local k_size    = {             7, 5,  5,  3,  3,  3,  3,  3,  3,   3,   3,  3,  3,  3,  3, 5 } -- kernel size


local function createModel(opt)
    
    local input = nn.Identity()()
    local Conv1 =  input
                - Conv(f_depths[1],f_depths[2],k_size[1],k_size[1],2,2,math.floor(k_size[1]/2),math.floor(k_size[1]/2))
                -- - MaxPool(2,2)
                - ReLU()
    local Conv2 =  Conv1
                - Conv(f_depths[2],f_depths[3],k_size[2],k_size[2],2,2,math.floor(k_size[2]/2),math.floor(k_size[2]/2))
                -- - MaxPool(2,2)
                - ReLU()
    local Conv3 =  Conv2
                - Conv(f_depths[3],f_depths[4],k_size[3],k_size[3],2,2,math.floor(k_size[3]/2),math.floor(k_size[3]/2))
                -- - MaxPool(2,2)
                - ReLU()
    local Conv4 =  Conv3
                - Conv(f_depths[4],f_depths[5],k_size[4],k_size[4],1,1,math.floor(k_size[4]/2),math.floor(k_size[4]/2))
                - ReLU()
    local Conv5 =  Conv4
                - Conv(f_depths[5],f_depths[6],k_size[5],k_size[5],2,2,math.floor(k_size[5]/2),math.floor(k_size[5]/2))
                -- - MaxPool(2,2)
                - ReLU()
    local Conv6 =  Conv5
                - Conv(f_depths[6],f_depths[7],k_size[6],k_size[6],1,1,math.floor(k_size[6]/2),math.floor(k_size[6]/2))
                - ReLU()
    local Conv7 =  Conv6
                - Conv(f_depths[7],f_depths[8],k_size[7],k_size[7],2,2,math.floor(k_size[7]/2),math.floor(k_size[7]/2))
                -- - MaxPool(2,2)
                - ReLU()
    local Conv8 =  Conv7
                - Conv(f_depths[8],f_depths[9],k_size[8],k_size[8],1,1,math.floor(k_size[8]/2),math.floor(k_size[8]/2))
                - ReLU()
    local Conv9 =  Conv8
                - Conv(f_depths[9],f_depths[10],k_size[9],k_size[9],2,2,math.floor(k_size[9]/2),math.floor(k_size[9]/2))
                -- - MaxPool(2,2)
                - ReLU()

    local Conv10 =  Conv9
                - Conv(f_depths[10],f_depths[11],k_size[10],k_size[10],1,1,math.floor(k_size[10]/2),math.floor(k_size[10]/2))
                - ReLU()

    local deConv0       = Conv10
                        - deConv(f_depths[11],f_depths[12],k_size[11],k_size[11],2,2,1,1,1,1)
                        - ReLU()
    
    local joinedFeat1   = { Conv8 , deConv0 }
                        - nn.JoinTable(2)
    local sideOutput1   = joinedFeat1
                        - UpSample(2)
                        - Conv(f_depths[12]+f_depths[9],2,5,5,1,1,2,2)
    local deConv1       = joinedFeat1
                        - deConv(f_depths[12]+f_depths[9],f_depths[13],k_size[12],k_size[12],2,2,1,1,1,1)
                        - ReLU()

    local joinedFeat2   = { Conv6 , deConv1 , sideOutput1 }
                        - nn.JoinTable(2)
    local sideOutput2   = joinedFeat2
                        - UpSample(2)
                        - Conv(f_depths[13]+f_depths[7]+2,2,5,5,1,1,2,2)
    local deConv2       = joinedFeat2
                        - deConv(f_depths[13]+f_depths[7]+2,f_depths[14],k_size[13],k_size[13],2,2,1,1,1,1)
                        - ReLU()

    local joinedFeat3   = { Conv4 , deConv2 , sideOutput2 }
                        - nn.JoinTable(2)
    local sideOutput3   = joinedFeat3
                        - UpSample(2)
                        - Conv(f_depths[14]+f_depths[5]+2,2,5,5,1,1,2,2)
    local deConv3       = joinedFeat3
                        - deConv(f_depths[14]+f_depths[5]+2,f_depths[15],k_size[14],k_size[14],2,2,1,1,1,1)
                        - ReLU()

    local joinedFeat4   = { Conv2 , deConv3 , sideOutput3 }
                        - nn.JoinTable(2)
    local sideOutput4   = joinedFeat4
                        - UpSample(2)
                        - Conv(f_depths[15]+f_depths[3]+2,2,5,5,1,1,2,2)
    local deConv4       = joinedFeat4
                        - deConv(f_depths[15]+f_depths[3]+2,f_depths[16],k_size[15],k_size[15],2,2,1,1,1,1)
                        - ReLU()                        

    local joinedFeat5   = { Conv1 , deConv4 , sideOutput4 }
                        - nn.JoinTable(2)
    -- sideOutput5 is the actual output of the network                        
    local sideOutput5   = joinedFeat5
                        - UpSample(2)
                        - Conv(f_depths[16]+f_depths[2]+2,f_depths[17],5,5,1,1,2,2)


   local model = nn.gModule({input} , {sideOutput5})     

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