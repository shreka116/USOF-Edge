--
--  Copyright (c) 2016, Computer Vision Lab @ Seoul National University.
--  All rights reserved.
--
--  Downloads data from web and save them as .t7 extensions
--

local image = require 'image'
local paths = require 'paths'
local ffi = require 'ffi'

local M = {}

local data_URL  = 'http://kitti.is.tue.mpg.de/kitti/data_stereo_flow.zip'

function M.exec(opt, cacheFile)
    if not paths.dirp(opt.genData .. '/KITTI_2012') then
        print("=> Downloading KITTI 2012 dataset from " .. data_URL)
        local down_ok   = os.execute('wget -P ' .. opt.genData .. '/ ' .. data_URL)
        assert(down_ok == true or down_ok == 0, 'error downloading KITTI 2012')
        local unzip_ok  = os.execute('unzip ' .. opt,genData .. '/data_stereo_flow.zip')
        assert(unzip_ok == true or unzip_ok == 0, 'error extracting data_stereo_flow.zip')
    end

    local trainDir      = paths.concat(opt.genData , 'KITTI_2012' , 'training', 'colored_0')
    local trainFlowDir  = paths.concat(opt.genData , 'KITTI_2012' , 'training', 'flow_noc')
    local valDir   = paths.concat(opt.genData , 'KITTI_2012' , 'testing', 'colored_0')

    local trainMaxLength         = math.max(-1, #(trainDir .. '/xxxxxx_xx.png')+1)
    local trainFlowMaxLength     = math.max(-1, #(trainFlowDir .. '/xxxxxx_xx.png')+1)
    local valMaxLength          = math.max(-1, #(valDir .. '/xxxxxx_xx.png')+1)

    local train_imagePath   = torch.CharTensor(194, 2, trainMaxLength)
    local train_flowPath    = torch.CharTensor(194, trainFlowMaxLength)
    local val_imagePath     = torch.CharTensor(195, 2, valMaxLength)
    
    local trainFiles    = paths.dir(trainDir)
    local trainFlowFiles= paths.dir(trainFlowDir)
    local valFiles    = paths.dir(valDir)

    table.sort(trainFiles)
    table.sort(trainFlowFiles)
    table.sort(valFiles)

    local tr_cnt        = 1
    local vl_cnt        = 1
    local counter       = 1


    for iter = 1, (#trainFiles-2)/2 do
        ffi.copy(train_imagePath[{ {tr_cnt},{1},{} }]:data(), paths.concat(trainDir, trainFiles[2*iter + 1]))
        ffi.copy(train_imagePath[{ {tr_cnt},{2},{} }]:data(), paths.concat(trainDir, trainFiles[2*iter + 2]))
        ffi.copy(train_flowPath[{ {tr_cnt},{} }]:data(), paths.concat(trainFlowDir, trainFlowFiles[iter + 2]))

        tr_cnt = tr_cnt + 1
    end

    for iter = 1, (#valFiles-2)/2 do
        ffi.copy(val_imagePath[{ {vl_cnt},{1},{} }]:data(), paths.concat(trainDir, trainFiles[2*iter + 1]))
        ffi.copy(val_imagePath[{ {vl_cnt},{2},{} }]:data(), paths.concat(trainDir, trainFiles[2*iter + 2]))

        vl_cnt = vl_cnt + 1
    end

    local datasetInfo = {
        train   =   {
            imagePath   =   train_imagePath,
            imageFlow   =   train_flowPath,
        },
        val     =   {
            imagePath   =   val_imagePath,
        },
    }

    print(" | saving list of FlyingChairs dataset to " .. cacheFile)
    torch.save(cacheFile, datasetInfo)
    return datasetInfo    
end

return M