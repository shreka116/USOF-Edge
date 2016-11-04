require 'torch'
require 'math'
require 'paths'

TAG_FLOAT = 202021.25

function readPNGFlowFile(filename)
---------------------------------------------
-- filename : full directory of the .png file
-- reads .png flow file
-- it only reads single png flow file
-- when using batch of files
-- use iterative loops to read each png flow files

    assert(paths.filep(filename), 'png file does not exist !!')
    assert(filename:sub(-4) == '.png', 'extension has to be .png flow file !!')

    local img = image.load(filename):double()
    local F_u = img[{ {1},{},{} }]:csub(torch.pow(2,15)):div(64)
    local F_v = img[{ {2},{},{} }]:csub(torch.pow(2,15)):div(64)    
    local F_valid = img[{ {3},{},{} }]:maskedFill(img[{ {3},{},{} }]:ge(1), 1)
    local invalid_mask = F_valid:eq(0)
    F_u:maskedFill(invalid_mask, 0)
    F_v:maskedFill(invalid_mask, 0)

    local outs = torch.zeros(img:size())
    outs[{ {1},{},{} }] = F_u
    outs[{ {2},{},{} }] = F_v
    outs[{ {3},{},{} }] = F_valid


    return outs
end


function readFlowFile(filename)
---------------------------------------------
-- filename : full directory of the .flo file
-- reads .flo file
-- it only reads single flo file
-- when using batch of files
-- use iterative loops to read each flo files


    assert(paths.filep(filename), 'flo file does not exist !!')
    assert(filename:sub(-4) == '.flo', 'extension has to be .flo !!')

    local ff = torch.DiskFile(filename,'r')
    ff:binary()
    local tag = ff:readFloat()

    if tag ~= TAG_FLOAT then
        print('unable to read file... perhaps bigendian error !!!')
    end
    
    local w = ff:readInt()
    local h = ff:readInt()
    local nbands = 2

    local tf = torch.FloatTensor(nbands*h*w)
    ff:readFloat(tf:storage())
    tf = tf:reshape(h, w*nbands)
    local flow = torch.FloatTensor(nbands+1, h, w)
    local evenarray = torch.range(2,2*w,2):long()
    local oddarray  = torch.range(1,2*w,2):long()

    flow[{ {1},{},{} }] = (tf:index(2,oddarray)):reshape(1,h,w)
    flow[{ {2},{},{} }] = (tf:index(2,evenarray)):reshape(1,h,w)
    flow[{ {3},{},{} }]:fill(1)
--    for i = 1, h do
--        for j = 1, w do
--            flow[{ {1},{i},{j} }] = tf[{ {i},{j} }]

--        end
--    end
    
--    flow[{ {1},{},{} }] = tf[{ {},{1,w*(nbands-1)} }]
--    flow[{ {2},{},{} }] = tf[{ {},{w*(nbands-1)+1,w*nbands} }]

    ff:close()    
    
    return flow
end

function evaluateEPE(est_flow, gt_flow, motion_pixel)

    local epe		= torch.CudaTensor(gt_flow:size())
    local epe_u 	= torch.csub(gt_flow[{ {},{1},{},{} }], est_flow[{ {},{2},{},{} }])
    local epe_v 	= torch.csub(gt_flow[{ {},{2},{},{} }], est_flow[{ {},{1},{},{} }])
    epe   		    = torch.sqrt(torch.add(epe_u:pow(2), epe_v:pow(2)))
    epe:maskedFill(gt_flow[{ {},{3},{},{} }]:eq(0), 0)

    local E               = epe:sum() / ( gt_flow[{ {},{3},{},{} }]:eq(1):sum() )
    local erroneous_pix   = ( epe:gt(motion_pixel):sum() ) / ( gt_flow[{ {},{3},{},{} }]:eq(1):sum() ) 

    return E , erroneous_pix
end

--[[
function evaluateEPE(est_flow, gt_flow, motion_pixel)

    local epe		= torch.CudaTensor(gt_flow:size())
    local epe_u 	= torch.csub(gt_flow[{ {},{1},{},{} }], est_flow[{ {},{2},{},{} }])
    local epe_v 	= torch.csub(gt_flow[{ {},{2},{},{} }], est_flow[{ {},{1},{},{} }])
    epe   		    = torch.sqrt(torch.add(epe_u:pow(2), epe_v:pow(2)))
--	print(epe:size())
    epe:maskedFill(gt_flow[{ {},{3},{},{} }]:eq(0), 0)
--    epe 		= ( epe:gt(motion_pixel):sum() ) / ( (gt_flow[{ {3},{},{} }]:eq(1)):sum() )

    local epe_less      = epe:le(motion_pixel)	
    local epe_under     = epe:maskedSelect(epe_less):sum() / epe_less:sum()

    if (epe_under ~= epe_under) then
        epe_under       =   1e6
    end
    -- local denominator   = epe_less:sum()
    -- local numerator     = epe:maskedSelect(epe_less):sum()
    -- if (denominator == 0) or (numerator == 0) then
	--     epe_under = 0
    -- else
    --     epe_under = numerator/denominator
    -- end

    local epe_greater   = epe:gt(motion_pixel)
    local epe_over      = epe:maskedSelect(epe_greater):sum() / epe_greater:sum() 

    if (epe_over ~= epe_over) then
        epe_over        =   1e6
    end
    -- denominator         = epe_greater:sum()
    -- numerator           = epe:maskedSelect(epe_greater):sum()
    -- if (denominator == 0) or (numerator == 0) then
	--     epe_over = 0
    -- else
    --     epe_over = numerator/denominator
    -- end

--    if (epe_over == epe_over) == false then
--	epe_over = 0
--    end

    local epe_total	= (epe_under + epe_over)
    if (epe_over == 1e6) or (epe_under == 1e6) then
        epe_total = 1e6
    end

    local erroneous_pixels = epe:gt(motion_pixel):sum() / gt_flow[{ {},{3},{},{} }]:eq(1):sum()
	
--    	return ( epe:gt(motion_pixel):sum() ) / ( (gt_flow[{ {3},{},{} }]:eq(1)):sum() )

    return epe_under, epe_over, epe_total, erroneous_pixels
end
]]--

function uvToColor(flow)
    local flow_size = flow:size()
    local unknown_flow_thresh = 1e9
    local unknown_flo         = 1e10
    local max_u               = -999
    local max_v               = -999
    local min_u               = 999
    local min_v               = 999
    local maxrad              = -1
--    local max_u               = torch.Tensor(flow_size[1], flow_size[2]):fill(-999)
--    local max_v               = torch.Tensor(flow_size[1], flow_size[2]):fill(-999)
--    local min_u               = torch.Tensor(flow_size[1], flow_size[2]):fill(999)
--    local min_v               = torch.Tensor(flow_size[1], flow_size[2]):fill(999)
--    local maxrad              = torch.Tensor(flow_size[1], 1):fill(-1)

    local rad                 = torch.Tensor(1, flow_size[2], flow_size[3]):fill(0)
    
    -- fix unknown flow
    local idxUnknown          = torch.abs(flow[{ {1},{},{} }]):gt(unknown_flow_thresh)
    local tmp                 = torch.abs(flow[{ {2},{},{} }]):gt(unknown_flow_thresh)
    idxUnknown:add(tmp)
  --  idxUnknown = idxUnknown:ge(1)
  
    flow[{ {1},{},{} }]:maskedFill(idxUnknown, 0)
    flow[{ {2},{},{} }]:maskedFill(idxUnknown, 0)
  
    tmp   = torch.max(torch.max(flow[{ {1},{},{} }],2),3):reshape(1,1)
    max_u = torch.cmax(tmp,max_u)
    tmp   = torch.max(torch.max(flow[{ {2},{},{} }],2),3):reshape(1,1)
    max_v = torch.cmax(tmp,max_v)
    
    min_u = torch.cmin(torch.min(torch.min(flow[{ {1},{},{} }],2),3):reshape(1,1),min_u)
    min_v = torch.cmin(torch.min(torch.min(flow[{ {2},{},{} }],2),3):reshape(1,1),min_v)
    
    rad   = torch.sqrt(torch.add(torch.pow(flow[{ {1},{},{} }],2),torch.pow(flow[{ {2},{},{} }],2)))
    maxrad= torch.cmax(torch.max(torch.max(rad,2),3):reshape(1,1), maxrad)
  
--   for i = 1, flow_size[1] do                                                     
        flow[{ {1},{},{} }] = torch.div(flow[{ {1},{},{} }],maxrad[1][1]+0.001)
        flow[{ {2},{},{} }] = torch.div(flow[{ {2},{},{} }],maxrad[1][1]+0.001) 
--    end
    
    -- compute color
    local img = computeColor(flow)

    -- unknown flow
    img[{ {1},{},{} }]:maskedFill(idxUnknown, 0)
    img[{ {2},{},{} }]:maskedFill(idxUnknown, 0)
    img[{ {3},{},{} }]:maskedFill(idxUnknown, 0)
    

    return img
end

function computeColor(flow)
    local flow_size = flow:size()
    local img       = torch.Tensor(3, flow_size[2], flow_size[3])
    
    if flow:type() == 'torch.CudaTensor' then
        img = img:cuda()
    end

    local nanIdx = torch.add(flow[{ {1},{},{} }]:ne(flow[{ {1},{},{} }]), flow[{ {2},{},{} }]:ne(flow[{ {2},{},{} }]))
    nanIdx = nanIdx:ge(1)
    flow[{ {1},{},{} }]:maskedFill(nanIdx,0)
    flow[{ {2},{},{} }]:maskedFill(nanIdx,0)
    -- need to implement
    


    local colorwheel        = makeColorwheel()
    if flow:type() == 'torch.CudaTensor' then
        colorwheel  = colorwheel:cuda()
    end

    local colorwheel_size   = colorwheel:size()

    local rad   = torch.sqrt(torch.add(torch.pow(flow[{ {1},{},{} }],2),torch.pow(flow[{ {2},{},{} }],2))):reshape(flow_size[2],flow_size[3]):t()
--    local a     = torch.atan2(-flow[{ {},{2},{},{} }],-flow[{ {},{1},{},{} }]):div(math.pi):reshape(flow_size[3],flow_size[4])
    local a     = torch.atan2(-flow[{ {2},{},{} }],-flow[{ {1},{},{} }]):div(math.pi):reshape(flow_size[2],flow_size[3]):t():reshape(flow_size[2]*flow_size[3])
    local fk    = ((torch.add(a,1):div(2)):mul(colorwheel_size[1]-1)):add(1) -- -1~1 maped to 1~ncols
    local k0    = torch.floor(fk)  -- 1, 2, ... , ncols
    local k1    = torch.add(k0,1)
    k1:maskedFill((torch.eq(k1,colorwheel_size[1]+1)),1)                                
   

    local f     = torch.csub(fk, k0):reshape(fk:size(1),1)

    for i = 1, colorwheel_size[2] do
        local tmp   = colorwheel[{ {},{i} }]
        local col0  = torch.div(tmp:index(1,k0:long()),255)
        local col1  = torch.div(tmp:index(1,k1:long()),255)
        

        local col   = ((1-f):cmul(col0)):add(torch.cmul(f,col1))

        local bool_idx      = torch.le(rad,1)
        local bool_idx_zero = bool_idx:eq(0)
        local tmp1          = (1 - torch.cmul(rad:maskedSelect(bool_idx), 1 - col:maskedSelect(bool_idx:reshape(flow_size[2]*flow_size[3])) ))
        col:maskedCopy(bool_idx, tmp1)

        local tmp2  = ( col:maskedSelect(bool_idx_zero:reshape(flow_size[2]*flow_size[3])) ):mul(0.75)
        col = col:maskedCopy(bool_idx_zero:reshape(flow_size[2]*flow_size[3]), tmp2):reshape(flow_size[3],flow_size[2]):t()
        img[{ {i},{},{} }] = torch.floor((1-nanIdx):float():cmul(col):mul(255))--:byte()
        
    end

    return img
end


function makeColorwheel()
    local RY = 15
    local YG = 6
    local GC = 4
    local CB = 11
    local BM = 13
    local MR = 6

    local ncols = RY+YG+GC+CB+BM+MR
    local colorwheel = torch.zeros(ncols, 3) -- r g b
    
    local col   = 0
  
    -- need to further implement
    -- RY
    colorwheel:sub(1,RY,1,1):fill(255)
--    colorwheel:sub(1,RY,2,2):fill(torch.floor(torch.range(0,RY-1):div(RY):mul(255)))                            
    colorwheel[{ {1,RY},{2,2} }] = torch.floor(torch.range(0,RY-1):div(RY):mul(255))
    col = col + RY

    -- YG
--    colorwheel:sub(1+col,YG+col,1,1):fill(torch.floor(torch.range(0,YG-1):div(YG):mul(255)))
    colorwheel[{ {1+col,YG+col},{1,1} }] = 255 - torch.floor(torch.range(0,YG-1):div(YG):mul(255))
    colorwheel:sub(1+col,YG+col,2,2):fill(255)                            
    col = col + YG
    
    -- GC
    colorwheel:sub(1+col,GC+col,2,2):fill(255)
--    colorwheel:sub(1+col,GC+col,3,3):fill(torch.floor(torch.range(0,GC-1):div(GC):mul(255)))                            
    colorwheel[{ {1+col,GC+col},{3,3} }] = torch.floor(torch.range(0,GC-1):div(GC):mul(255))
    col = col + GC

    -- CB
--    colorwheel:sub(1+col,CB+col,2,2):fill(torch.floor(torch.range(0,CB-1):div(CB):mul(255)))
    colorwheel[{ {1+col,CB+col},{2,2} }] = 255 - torch.floor(torch.range(0,CB-1):div(CB):mul(255))
    colorwheel:sub(1+col,CB+col,3,3):fill(255)                            
    col = col + CB

    -- BM
    colorwheel:sub(1+col,BM+col,3,3):fill(255)
--    colorwheel:sub(1+col,BM+col,1,1):fill(torch.floor(torch.range(0,BM-1):div(BM):mul(255)))                            
    colorwheel[{ {1+col,BM+col},{1,1} }] = torch.floor(torch.range(0,BM-1):div(BM):mul(255))
    col = col + BM

    -- MR
--    colorwheel:sub(1+col,MR+col,3,3):fill(torch.floor(torch.range(0,MR-1):div(MR):mul(255)))
    colorwheel[{ {1+col,MR+col},{3,3} }] = 255 - torch.floor(torch.range(0,MR-1):div(MR):mul(255))
    colorwheel:sub(1+col,MR+col,1,1):fill(255)                            
    


    return colorwheel
end    


function flowToColor(flow)
    flow = flow:double()

    local F_du      = flow[{ {1},{},{} }]:clone()
    local F_dv      = flow[{ {2},{},{} }]:clone()
    local F_val     = flow[{ {3},{},{} }]:clone()
    local u_max     = F_du:maskedSelect(F_val:byte()):abs():max()
    local v_max     = F_dv:maskedSelect(F_val:byte()):abs():max()
    local F_max     = math.max(u_max, v_max)
    local F_mag     = torch.sqrt(torch.pow(F_du,2):add(torch.pow(F_dv,2)))                                            
    local F_dir     = torch.atan2(F_dv, F_du)
    local img       = flow_map(F_mag, F_dir, F_val, F_max, 8)
    
    return img
end

function flow_map(F_mag, F_dir, F_val, F_max, n)
    local img_size  = F_mag:size()
    local img       = torch.DoubleTensor(3, img_size[2], img_size[3]):fill(0)
    
    img[{ {1},{},{} }]  = torch.remainder(F_dir:div(2*math.pi),1)
    img[{ {2},{},{} }]  = F_mag:mul(n/F_max)
    img[{ {3},{},{} }]  = n - img[{ {2},{},{} }]
    img[{ {2,3},{},{} }]= torch.cmin(torch.cmax(img[{ {2,3},{},{} }], 0), 1)
    img                 = image.hsv2rgb(img)
    img[{ {1},{},{} }]  = img[{ {1},{},{} }]:cmul(F_val)
    img[{ {2},{},{} }]  = img[{ {2},{},{} }]:cmul(F_val)
    img[{ {3},{},{} }]  = img[{ {3},{},{} }]:cmul(F_val)

    return img
end

function writeFlowFile(flow, filename)
    
    if paths.filep(filename) then
        errpr(filename.. 'exists!!')
    end

    if not filename:match('.flo$') then
        filename = filename..'.flo'
    end

    local ff = torch.DiskFile(filename, 'w')
    ff:binary()
    ff:writeFloat(TAG_FLOAT)

    local w         = flow:size(3)
    local h         = flow:size(2)
    local nbands    = flow:size(1)
    if nbands > 2 then
--        print('only writing first 2 slices in 3 dim of flow')
        nbands = 2
    elseif nbands < 2 then
        error('flow tensor must hace x and y values format [2][h][w]')
    end
    ff:writeInt(w)
    ff:writeInt(h)
    local tt = torch.FloatTensor(nbands,w,h)
--    print(tt:size())
--    print(flow:size())
    tt:select(1,1):copy(flow:select(1,3))
    tt:select(1,2):copy(flow:select(1,2)) 
    ff:writeFloat(tt:storage())
    ff:close()
    return 1

end
