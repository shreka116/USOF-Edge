--
--  Copyright (c) 2016, Computer Vision Lab @ Seoul National University.
--  All rights reserved.
--
--  Downloads data from web and save them as .t7 extensions
--

local M = {}

local data_URL  = 'http://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs/FlyingChairs.zip'
local split_URL = 'http://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs/FlyingChairs_train_val.txt'


function M.exec(opt, cacheFile)
    if not paths.dirp(opt.genData .. '/FlyingChairs_release') then
        print("=> Downloading FlyingChairs dataset from " .. data_URL)
        local down_ok   = os.execute('wget -P ' .. opt.genData .. '/ ' .. data_URL)
        assert(down_ok == true or down_ok == 0, 'error downloading FlyingChairs')
        local unzip_ok  = os.execute('unzip ' .. opt,genData .. '/FlyingChairs.zip')
        assert(unzip_ok == true or unzip_ok == 0, 'error extracting FlyingChairs.zip')
    end

    if not paths.filep(opt.genData .. '/FlyingChairs_release/FlyingChairs_train_val.txt') then
        print("=> Downloading FlyingChairs training and validation split list from " .. split_URL)
        local down_ok   = os.execute('wget -P ' .. opt.genData .. '/FlyingChairs_release/ ' .. split_URL)
        assert(down_ok == true or down_ok == 0, 'error downloading train-validation split list')
    end
    
    local tr_lv_split   = io.open('FlyingChairs_train_val.txt','r')
    local dir           = paths.dir(opt.genData .. '/FlyingChairs_release/data/')
    table.sort(dir)

    local train_pair    = torch.CudaTensor(22232, 6, 384, 512)  -- previously counted
    local train_flow    = torch.CudaTensor(22232, 2, 384, 512)
    local val_pair      = torch.CudaTensor(640, 6, 384, 512)    -- previously counted
    local val_flow      = torch.CudaTensor(640, 2, 384, 512)

    local tr_cnt        = 1
    local vl_cnt        = 1
    local counter       = 1
    for line in tr_vl_split:lines() do
        if line == '1' then     -- train data
            train_pair[{ {tr_cnt},{1,3},{},{} }]    = image.load(opt.genData .. '/FlyingChairs_release/data/' .. dir[counter*3 + 1])  -- loading reference frame
            train_pair[{ {tr_cnt},{4,6},{},{} }]    = image.load(opt.genData .. '/FlyingChairs_release/data/' .. dir[counter*3 + 2])  -- loading target frame
            train_flow[{ {tr_cnt},{},{},{} }]       = readFlowFile(opt.genData .. '/FlyingChairs_release/data/' .. dir[counter*3])    -- loading ground-truth flow  

            tr_cnt = tr_cnt + 1
        elseif line == '2' then -- validation data
            val_pair[{ {vl_cnt},{1,3},{},{} }]      = image.load(opt.genData .. '/FlyingChairs_release/data/' .. dir[counter*3 + 1])
            val_pair[{ {vl_cnt},{4,6},{},{} }]      = image.load(opt.genData .. '/FlyingChairs_release/data/' .. dir[counter*3 + 2])
            val_flow[{ {vl_cnt},{},{},{} }]         = readFlowFile(opt.genData .. '/FlyingChairs_release/data/' .. dir[counter*3])

            vl_cnt = vl_cnt + 1
        end
        counter = counter + 1
    end


    print(" | saving FlyingChairs dataset to " .. cacheFile)
    torch.save(cacheFile, {
        train   = {
            data    = train_pair,
            labels  = train_flow,
        },
        val     = {
            data    = val_pair,
            labels  = val_flow,
        },
    })
    
end

return M