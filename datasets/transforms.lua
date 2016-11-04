--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Image transforms for data augmentation and input normalization
--

require 'image'

local M = {}

function M.SelectTransform(transforms)
   return function(input)
      local trTypes     = #transforms-- + 1           -- added one for identity (w/o transformation)
      local relative_trans
      local photometric = torch.random(1,6)
      local geometric   = torch.random(6,10)

      if trTypes ~= 1 then
        input  = transforms[photometric](input)
        input  = transforms[geometric](input)

        local relative_trans = torch.random(1,3)
        local rel_trans = torch.random(11,13)
      
        if relative_trans == 1 then
          input[{ {1,3},{},{} }] = transforms[rel_trans](input[{ {1,3},{},{} }])
        elseif relative_trans == 2 then
          input[{ {4,6},{},{} }] = transforms[rel_trans](input[{ {4,6},{},{} }])
        elseif relative_trans == 3 then 

        end      

      end

      return input
   end
end


-- function M.SelectTransform(transforms)
--    return function(input)
--       local trTypes     = #transforms-- + 1           -- added one for identity (w/o transformation)
--       local relative_trans
--       local photometric = torch.random(1,6)
--       local geometric   = torch.random(6,10)

--       if trTypes ~= 1 then
--         input  = transforms[photometric](input)
--         input  = transforms[geometric](input)
--       end

--       return input
--    end
-- end



function M.Compose(transforms)
   return function(input)
      for idx, transform in ipairs(transforms) do
        --  print(tostring(idx) .. '-->' .. tostring(transform))
         input = transform(input)
        --  print(tostring(idx) .. '-input -->' .. tostring(input:size()))
      end
    --   print('outa for loop-input -->' .. tostring(input:size()))
      return input
   end
end

function M.ColorNormalize(meanstd)
   return function(img)
      img = img:clone()
      for i=1,3 do
         img[i]:add(-meanstd.mean[i])
         img[i]:div(meanstd.std[i])
      end
      return img
   end
end


-- Lighting noise (AlexNet-style PCA-based noise)
function M.Lighting(alphastd, eigval, eigvec)
   return function(input)
      if alphastd == 0 then
         return input
      end

      local alpha = torch.Tensor(3):normal(0, alphastd)
      local rgb = eigvec:clone()
         :cmul(alpha:view(1, 3):expand(3, 3))
         :cmul(eigval:view(1, 3):expand(3, 3))
         :sum(2)
         :squeeze()

      input = input:clone()
      for i=1,3 do
         input[i]:add(rgb[i])
      end
      return input
   end
end

local function blend(img1, img2, alpha)
   return img1:mul(alpha):add(1 - alpha, img2)
end

local function grayscale(dst, img)
   dst:resizeAs(img)
   dst[1]:zero()
   dst[1]:add(0.299, img[1]):add(0.587, img[2]):add(0.114, img[3])
   dst[2]:copy(dst[1])
   dst[3]:copy(dst[1])
   return dst
end

function M.Saturation(var)
   local gs

   return function(input)
      gs = gs or input.new()
      grayscale(gs, input)

      local alpha = 1.0 + torch.uniform(-var, var)
      blend(input, gs, alpha)
      return input
   end
end

function M.Brightness(var)
   local gs

   return function(input)
      gs = gs or input.new()
      gs:resizeAs(input):zero()

      local alpha = 1.0 + torch.uniform(-var, var)
      blend(input, gs, alpha)
      return input
   end
end


function M.AdditiveGausNoise(var_1, var_2)

    return function(input)
        local gs = input.new()
        gs:resizeAs(input):zero()
        -- print(gs:size())

        local sigma = torch.uniform(var_1, var_2)
        torch.normal(gs:select(1,1), 0, sigma)
            gs:select(1,2):copy(gs:select(1,1))
            gs:select(1,3):copy(gs:select(1,1))
            gs:select(1,4):copy(gs:select(1,1))
            gs:select(1,5):copy(gs:select(1,1))
            gs:select(1,6):copy(gs:select(1,1))
       
        return input:add(gs)
    end
end

function M.Contrast(var_1, var_2)

   return function(input)
      local gs = input.new()
      gs:resizeAs(input):zero()
    --   local ref_gray = rgb2gray(input[{ {1,3},{},{} }])
    --   local tar_gray = rgb2gray(input[{ {4,6},{},{} }])
      grayscale(gs[{ {1,3},{},{} }], input[{ {1,3},{},{} }])
      grayscale(gs[{ {4,6},{},{} }], input[{ {4,6},{},{} }])
      gs[{ {1,3},{},{} }]:fill(gs[{ {1,3},{},{} }][1]:mean())
      gs[{ {4,6},{},{} }]:fill(gs[{ {4,6},{},{} }][1]:mean())

      local alpha = 1.0 + torch.uniform(var_1, var_2)
      blend(input, gs, alpha)
      return input
   end
end

function M.MultiplicativeColorChange(var_1, var_2)

    return function(input)

      local mult_R = torch.uniform(var_1, var_2)
      local mult_G = torch.uniform(var_1, var_2)
      local mult_B = torch.uniform(var_1, var_2)

      input:select(1,1):mul(mult_R)
      input:select(1,2):mul(mult_G)
      input:select(1,3):mul(mult_B)
      input:select(1,4):mul(mult_R)
      input:select(1,5):mul(mult_G)
      input:select(1,6):mul(mult_B)


      return input
    end
end

function M.AdditiveBrightness(var)

    return function(input) 

      local ref_hsl = image.rgb2hsl(input[{ {1,3},{},{} }])
      local tar_hsl = image.rgb2hsl(input[{ {4,6},{},{} }])
      local changes = torch.normal(0, 0.2)
      ref_hsl:select(1,3):add(changes)
      tar_hsl:select(1,3):add(changes)
      input[{ {1,3},{},{} }]:copy(image.hsl2rgb(ref_hsl))
      input[{ {4,6},{},{} }]:copy(image.hsl2rgb(tar_hsl))

      return input
    end
end

function M.GammaChanges(var_1, var_2)

    return function(input) 
      local ref_hsl = image.rgb2hsl(input[{ {1,3},{},{} }])
      local tar_hsl = image.rgb2hsl(input[{ {4,6},{},{} }])
      local gamma   = torch.uniform(var_1, var_2)
      ref_hsl:select(1,3):pow(gamma)
      tar_hsl:select(1,3):pow(gamma)
      input[{ {1,3},{},{} }]:copy(image.hsl2rgb(ref_hsl))
      input[{ {4,6},{},{} }]:copy(image.hsl2rgb(tar_hsl))

      return input
    end
end

-- function M.Translations(var)
    
--     return function(input)
--        local inputSize = input:size()
--        local trans_x = torch.random(-var*inputSize[3], var*inputSize[3])
--        local trans_y = torch.random(-var*inputSize[3], var*inputSize[3])

--        local x_from, x_to, y_from, y_to = 0,0,0,0

--        if (trans_x < 0) and (trans_y < 0) then
--           x_from  = 1
--           x_to    = inputSize[3] + trans_x
--           y_from  = 1
--           y_to    = inputSize[2] + trans_y
--        elseif (trans_x < 0) and (trans_y >= 0) then
--           x_from  = 1
--           x_to    = inputSize[3] + trans_x
--           y_from  = trans_y
--           y_to    = inputSize[2]
--        elseif (trans_x >= 0) and (trans_y < 0) then
--           x_from  = trans_x
--           x_to    = inputSize[3]
--           y_from  = 1
--           y_to    = inputSize[2] + trans_y
--        elseif (trans_x >= 0) and (trans_y >= 0) then
--           x_from  = trans_x
--           x_to    = inputSize[3]
--           y_from  = trans_y
--           y_to    = inputSize[2]
--        end
       
--        local preDefinedSz = {256, 384}
--        local rndx   = torch.random(x_from, x_to - preDefinedSz[2])
--        local rndy   = torch.random(y_from, y_to - preDefinedSz[1])
--        local rndx2  = torch.random(x_from, x_to - preDefinedSz[2])
--        local rndy2  = torch.random(y_from, y_to - preDefinedSz[1])
--         -- channels x 256 x 384 image
--       --   print(x_from,x_to,y_from,y_to)
--       --  print(rndx,rndy, rndx2, rndy2)
       

--        local translated     = image.crop(input, rndx, rndy, rndx + preDefinedSz[2], rndy + preDefinedSz[1])
--        local rel_translated = image.crop(input, rndx2, rndy2, rndx2 + preDefinedSz[2], rndy2 + preDefinedSz[1])
--       --  print('tranlsation')
--       --  print(translated:size())
--       --  print(rel_translated:size())
--       local rel_trans = torch.random(3) -- if 1 replace ref. image, if 2 replace target image, if 3 no change
--       if rel_trans == 1 then
--         translated[{ {1,3},{},{} }] = rel_translated[{ {1,3},{},{} }]:clone()
--       elseif rel_trans == 2 then
--         translated[{ {4,6},{},{} }] = rel_translated[{ {4,6},{},{} }]:clone()
--       end



--        return translated--, rel_translated
--     end
-- end


function M.Translations_naive(var)  
    return function(input)
       local inputSize = input:size()
       local trans_x = torch.random(-var*inputSize[3], var*inputSize[3])
       local trans_y = torch.random(-var*inputSize[3], var*inputSize[3])

      --  local x_from, x_to, y_from, y_to = 0,0,0,0


      --  if (trans_x <= 0) and (trans_y <= 0) then
      --     x_from  = 1
      --     x_to    = inputSize[3] + trans_x
      --     y_from  = 1
      --     y_to    = inputSize[2] + trans_y
      --  elseif (trans_x <= 0) and (trans_y > 0) then
      --     x_from  = 1
      --     x_to    = inputSize[3] + trans_x
      --     y_from  = trans_y
      --     y_to    = inputSize[2]
      --  elseif (trans_x > 0) and (trans_y <= 0) then
      --     x_from  = trans_x
      --     x_to    = inputSize[3]
      --     y_from  = 1
      --     y_to    = inputSize[2] + trans_y
      --  elseif (trans_x > 0) and (trans_y > 0) then
      --     x_from  = trans_x
      --     x_to    = inputSize[3]
      --     y_from  = trans_y
      --     y_to    = inputSize[2]
      --  end


      --  print('translations')
      -- --  print(image.crop(input, x_from, y_from, x_to, y_to):size())

      -- --  return image.crop(input, x_from, y_from, x_to, y_to)
      -- local translated_out = torch.zeros(input:size())
      --  return translated_out[{ {},{y_from,y_to},{x_from,x_to} }]
        return image.translate(input, trans_x, trans_y)
    end
end

function M.Translations_wo_blacks(var)  
    return function(input)
       local inputSize = input:size()
       local trans_x = torch.random(-var*inputSize[3], var*inputSize[3])
       local trans_y = torch.random(-var*inputSize[3], var*inputSize[3])

       local x_from, x_to, y_from, y_to = 0,0,0,0


       if (trans_x <= 0) and (trans_y <= 0) then
          x_from  = 1
          x_to    = inputSize[3] + trans_x
          y_from  = 1
          y_to    = inputSize[2] + trans_y
       elseif (trans_x <= 0) and (trans_y > 0) then
          x_from  = 1
          x_to    = inputSize[3] + trans_x
          y_from  = trans_y
          y_to    = inputSize[2]
       elseif (trans_x > 0) and (trans_y <= 0) then
          x_from  = trans_x
          x_to    = inputSize[3]
          y_from  = 1
          y_to    = inputSize[2] + trans_y
       elseif (trans_x > 0) and (trans_y > 0) then
          x_from  = trans_x
          x_to    = inputSize[3]
          y_from  = trans_y
          y_to    = inputSize[2]
       end


      --  print('translations')
      -- --  print(image.crop(input, x_from, y_from, x_to, y_to):size())

       return image.crop(input, x_from, y_from, x_to, y_to)
      -- local translated_out = torch.zeros(input:size())
      --  return translated_out[{ {},{y_from,y_to},{x_from,x_to} }]
        -- return image.translate(input, trans_x, trans_y)
    end
end

function M.HorizontalFlip(prob)
   return function(input)
      if torch.uniform() < prob then
       input = image.hflip(input)
      end
      return input
   end
end

-- function M.Rotation(var)
--    return function(input)
--      local deg      = torch.uniform(-var, var)
--      local rotated  = image.rotate(input, deg * math.pi / 180, 'bilinear')
--      local center_x = math.ceil(input:size(3)/2)
--      local center_y = math.ceil(input:size(2)/2)

--      local rel_deg      = torch.uniform(deg - 5, deg + 5)
--      local rel_rotated  = image.rotate(input, rel_deg * math.pi / 180, 'bilinear')
     
--      local rotated_input = image.crop(rotated, center_x - 192, center_y - 128, center_x + 192, center_y + 128)
--      local rel_rotated   = image.crop(rel_rotated, center_x - 192, center_y - 128, center_x + 192, center_y + 128)
--     local rel_trans = torch.random(3) -- if 1 replace ref. image, if 2 replace target image, if 3 no change
--     if rel_trans == 1 then
--       rotated_input[{ {1,3},{},{} }] = rel_rotated[{ {1,3},{},{} }]:clone()
--     elseif rel_trans == 2 then
--       rotated_input[{ {4,6},{},{} }] = rel_rotated[{ {4,6},{},{} }]:clone()
--     end


--      return rotated_input
--    end
-- end

function M.Rotation_naive(var)
   return function(input)
     local deg      = torch.uniform(-var, var)
    --  local outs     = image.rotate(input, deg * math.pi / 180, 'bilinear')
    --  print(outs:size())
    --  print('rotations')
    --  print(image.rotate(input, deg * math.pi / 180, 'bilinear'):size())
     return image.rotate(input, deg * math.pi / 180, 'bilinear')
   end
end

function M.Rotation_wo_blacks_init(var)
   return function(input)
     local deg      = torch.uniform(-var, var)
     local h,w      = input:size(2),input:size(3)
     local rot_img  = image.rotate(input, deg * math.pi / 180, 'bilinear')

    --  local outs     = image.crop(rot_img, w/2 - 200, h/2, - 139, w/2 + 200, h/2 + 139)


     return image.crop(rot_img, w/2 - 200, h/2 - 139, w/2 + 200, h/2 + 139)
   end
end

function M.Rotation_wo_blacks_relative(var)
   return function(input)
     local deg      = torch.uniform(-var, var)
     local h,w      = input:size(2),input:size(3)
     local rot_img  = image.rotate(input, deg * math.pi / 180, 'bilinear')

    --  local outs     = image.crop(rot_img, w/2 - 200, h/2, - 139, w/2 + 200, h/2 + 139)


     return image.crop(rot_img, 200 - 190, 139 - 134, 200 + 190, 139 + 134)
   end
end

-- function M.Scales(minSize, maxSize)
--    return function(input)
--       local w, h        = input:size(3), input:size(2)
--       local factors     = torch.uniform(minSize, maxSize)
--       local w1          = math.ceil(w*factors)
--       local h1          = math.ceil(h*factors)
--       local scaled      = image.scale(input, w1, h1)
      
--       local rel_factors = torch.uniform(factors + 0.1,factors + 0.3)
--       local rel_w       = math.ceil(w*rel_factors)
--       local rel_h       = math.ceil(h*rel_factors)
--       local rel_scaled  = image.scale(input, rel_w, rel_h)
      
--       local scaled_input, rel_scaled_input
--       local preDefinedSz = {256, 384}

--       if factors ~= 1 then

--         local center_x      = math.ceil(scaled:size(3)/2)
--         local center_y      = math.ceil(scaled:size(2)/2)
--         scaled_input        = image.crop(scaled, center_x - preDefinedSz[2]/2, center_y - preDefinedSz[1]/2, center_x + preDefinedSz[2]/2, center_y + preDefinedSz[1]/2)

--         local rel_center_x  = math.ceil(rel_scaled:size(3)/2)
--         local rel_center_y  = math.ceil(rel_scaled:size(2)/2)
--         rel_scaled_input    = image.crop(rel_scaled, rel_center_x - preDefinedSz[2]/2, rel_center_y - preDefinedSz[1]/2, rel_center_x + preDefinedSz[2]/2, rel_center_y + preDefinedSz[1]/2)

--       else

--         scaled_input        = input:clone()

--         local rel_center_x  = math.ceil(rel_scaled:size(3)/2)
--         local rel_center_y  = math.ceil(rel_scaled:size(2)/2)
--         rel_scaled_input    = image.crop(rel_scaled, rel_center_x - preDefinedSz[2]/2, rel_center_y - preDefinedSz[1]/2, rel_center_x + preDefinedSz[2]/2, rel_center_y + preDefinedSz[1]/2)
--       end


--         local rel_trans = torch.random(3) -- if 1 replace ref. image, if 2 replace target image, if 3 no change
--         if rel_trans == 1 then
--           scaled_input[{ {1,3},{},{} }] = rel_scaled_input[{ {1,3},{},{} }]:clone()
--         elseif rel_trans == 2 then
--           scaled_input[{ {4,6},{},{} }] = rel_scaled_input[{ {4,6},{},{} }]:clone()
--         end

--       return scaled_input 
--    end
-- end
function M.Scales_naive(minSize, maxSize)
   return function(input)
      local w, h        = input:size(3), input:size(2)
      local factors     = torch.uniform(minSize, maxSize)
      local w1          = math.ceil(w*factors)
      local h1          = math.ceil(h*factors)
      local scaled      = image.scale(input, w1, h1)
      local scaled_input= torch.zeros(input:size(1),h,w)


      if factors > 1 then

        local center_x      = math.ceil(w1/2)
        local center_y      = math.ceil(h1/2)
        scaled_input        = image.crop(scaled, center_x - w/2, center_y - h/2, center_x + w/2, center_y + h/2)

        -- print(center_x, center_y)
        -- print(scaled_input:size())
        -- print(w,h)
        -- print(w1,h1)
        -- print(factors)
      
      elseif factors < 1 then
  
        scaled_input[{ {},{1,h1},{1,w1} }] = scaled:clone()
  
      else
         
         scaled_input        = input:clone()

      end
            -- print('scales')

      -- print(scaled_input:size())
      return scaled_input 
   end
end

function M.Scales_wo_blacks(minSize, maxSize)
   return function(input)
      local w, h        = input:size(3), input:size(2)
      local factors     = torch.uniform(minSize, maxSize)
      local w1          = math.ceil(w*factors)
      local h1          = math.ceil(h*factors)
      local scaled      = image.scale(input, w1, h1)
      local scaled_input= torch.zeros(input:size(1),h,w)


      if factors > 1 then
        local x_from = torch.random(1, w1-w)
        local y_from = torch.random(1, h1-h)

        scaled_input = image.crop(scaled, x_from, y_from, x_from + w, y_from + h)

        -- local center_x      = math.ceil(w1/2)
        -- local center_y      = math.ceil(h1/2)
        -- scaled_input        = image.crop(scaled, center_x - w/2, center_y - h/2, center_x + w/2, center_y + h/2)

        -- print(center_x, center_y)
        -- print(scaled_input:size())
        -- print(w,h)
        -- print(w1,h1)
        -- print(factors)
      
      elseif factors < 1 then
  
        scaled_input = scaled:clone()
  
      else
         
         scaled_input        = input:clone()

      end
            -- print('scales')

      -- print(scaled_input:size())
      return scaled_input 
   end
end


function M.nCrop()
    return function(input)
       local inputSz     = input:size()
       if (inputSz[2] ~= inputSz[3]) then
          local imgSz       = math.min(inputSz[2] - 128,inputSz[3] - 128) --256
          local largeGaps   = math.floor(imgSz*0.4)                       --25
          local largeGaps_half = math.floor(largeGaps/2)                  --12
          local largeInputs = torch.zeros(3, imgSz + largeGaps, imgSz + largeGaps - 1) -- 3x280x280
          local rndPos_x    = torch.random(1 + largeGaps_half, inputSz[3] - largeGaps_half - imgSz) -- 13~244
          local rndPos_y    = torch.random(1 + largeGaps_half, inputSz[2] - largeGaps_half - imgSz) -- 13~116

          -- print ( rndPos_x - largeGaps_half,  rndPos_y - largeGaps_half,rndPos_x + (imgSz + largeGaps_half), rndPos_y + (imgSz + largeGaps_half))
          largeInputs       = image.crop(input, rndPos_x - largeGaps_half, rndPos_y - largeGaps_half, rndPos_x + (imgSz + largeGaps_half), rndPos_y + (imgSz + largeGaps_half))
          
          input             = image.crop(input, rndPos_x, rndPos_y, rndPos_x + imgSz, rndPos_y + imgSz)
          return input, largeInputs
       else
          return input
       end
    end
end

function M.randomCrop()
    return function(input)
       local inputSz     = input:size()
       
       local x_from      = torch.random(1,inputSz[3]-384)
       local y_from      = torch.random(1,inputSz[2]-256)

       return image.crop(input, x_from, y_from, x_from + 384, y_from + 256)
    end
end

function M.Identity()
   return function(input)
      return input
   end
end



return M
