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

local numChannels       = 3

local Conv      = nn.SpatialConvolution
local deConv    = nn.SpatialFullConvolution
local ReLU      = nn.ReLU
local MaxPool   = nn.SpatialMaxPooling
local UpSample  = nn.SpatialUpSamplingNearest

local f_depths  = {2*numChannels,64,128,256,256,512,512,512,512,1024,512,512,256,128,64,2} -- feature depths
local k_size    = { 7, 5,  5,  3,  3,  3,  3,  3,  3,   1,  1,  1,  1,  1, 1 } -- kernel size

local function createModel(opt)

   -- begin modelling USCNN
	local model             = nn.Sequential()
    -- local feature_depths    = {2*numChannels, 256, 256, 512, 512, 512 ,512, 1024, 512, 256, 128, 64, 2}
    local feature_depths    = {2*numChannels, 256/2, 256/2, 512/2, 512/2, 512/2 ,512/2, 1024/2, 512/2, 256/2, 128/2, 64/2, 2}
    local kernel_size       = {7, 5, 5, 3, 3, 3, 3, 1, 5, 5, 5 , 5}
    -- local kernel_size       = {7, 5, 5, 3, 3, 3, 3, 1, 5, 5, 5 , 13, 15}
    
    -- very first conv. layer
    model:add(nn.SpatialConvolution(feature_depths[1], feature_depths[1+1], kernel_size[1], kernel_size[1], 1, 1, math.floor(kernel_size[1]/2), math.floor(kernel_size[1]/2)) :init('weight', nninit.normal, 0.0, math.sqrt(2/(kernel_size[1]*kernel_size[1]*feature_depths[1])))
                                                                        :init('bias', nninit.constant, 0))
    model:add(nn.ReLU())

    -- intermediate conv. layers
    
    for i=2,#kernel_size-1 do
    --  for i=2,#kernel_size-2 do
        if i < 9 then -- downsamping layers
            if (i%2== 1) then -- apply pooling after odd-th conv. layer
                model:add(nn.SpatialConvolution(feature_depths[i], feature_depths[i+1], kernel_size[i], kernel_size[i], 1, 1, math.floor(kernel_size[i]/2), math.floor(kernel_size[i]/2)) :init('weight', nninit.normal, 0.0, math.sqrt(2/(kernel_size[i]*kernel_size[i]*feature_depths[i])))
                                                                                                                                                                                        :init('bias', nninit.constant, 0))	
            else    
                model:add(nn.SpatialConvolution(feature_depths[i], feature_depths[i+1], kernel_size[i], kernel_size[i], 2, 2, math.floor(kernel_size[i]/2), math.floor(kernel_size[i]/2)) :init('weight', nninit.normal, 0.0, math.sqrt(2/(kernel_size[i]*kernel_size[i]*feature_depths[i])))
                                                                                                                                                                                        :init('bias', nninit.constant, 0))
            end

            model:add(nn.ReLU())
        else -- upsampling layers
            model:add(nn.SpatialUpSamplingNearest(2))
            model:add(nn.SpatialConvolution(feature_depths[i], feature_depths[i+1], kernel_size[i], kernel_size[i], 1, 1, math.floor(kernel_size[i]/2), math.floor(kernel_size[i]/2)) :init('weight', nninit.normal, 0.0, math.sqrt(2/(kernel_size[i]*kernel_size[i]*feature_depths[i])))
                                                                                                                                                                                    :init('bias', nninit.constant, 0))
            model:add(nn.ReLU())
        end

    end

    model:add(nn.SpatialUpSamplingNearest(2))

    model:add(nn.SpatialConvolution(feature_depths[#kernel_size], feature_depths[#kernel_size+1], kernel_size[#kernel_size], kernel_size[#kernel_size], 1, 1, math.floor(kernel_size[#kernel_size]/2), math.floor(kernel_size[#kernel_size]/2)):init('weight', nninit.normal, 0.0, math.sqrt(2/(kernel_size[#kernel_size]*kernel_size[#kernel_size]*feature_depths[#kernel_size]))))
                                                                                   
    return model
end

return createModel