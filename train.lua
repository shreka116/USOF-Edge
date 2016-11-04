--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The training loop and learning rate schedule
--

local optim = require 'optim'

local M = {}
local Trainer = torch.class('USOF.Trainer', M)

function Trainer:__init(model, edge_model, warp_image, warp_edge, smoothness_loss, criterion, edge_loss, opt, optimState)
   self.model           = model
   self.edge_model      = edge_model
   self.warp_image      = warp_image
   self.warp_edge       = warp_edge
   self.smoothness_loss = smoothness_loss
   self.edge_loss       = edge_loss
   self.criterion       = criterion
   self.optimState      = optimState or {
      learningRate      = opt.learningRate,
      learningRateDecay = 0.0,
      beta1             = opt.beta_1,
      beta2             = opt.beta_2,
   }
  --  self.optimState      = optimState or {
  --     learningRate      = opt.learningRate,
  --     learningRateDecay = 0.0,
  --     momentum          = 0.9,
  --     nesterov          = true,
  --     dampening         = 0.0,
  --     weightDecay       = 5e-4,
  --  }
   self.opt = opt
   self.params, self.gradParams = model:getParameters()
end

function Trainer:train(epoch, dataloader)
   -- Trains the model for a single epoch
   self.optimState.learningRate = self:learningRate(epoch)
   local timer              = torch.Timer()
   local dataTimer          = torch.Timer()
   local criterion_output   = 0.0

   local function feval()
    --   return self.criterion.output, self.gradParams
      return criterion_output, self.gradParams
   end

   local trainSize  = dataloader:size()
   local lossSum    = 0.0
   local N          = 0
   local debug_loss = 0.0

   print('=============================')
   print(self.optimState)
   print('=============================')
   
   print('=> Training epoch # ' .. epoch)
   self.model:training()
   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)
      

      -- run edge detector
      self.edge_model:forward(self.edgeInput[{ {},{1,3},{},{} }])
      local ref_edges           = {
          { 
            1 - self.edge_model.output[1][1] ,
          1 - self.edge_model.output[1][2],
          1 - self.edge_model.output[1][3],
          1 - self.edge_model.output[1][4],
          1 - self.edge_model.output[1][5], 
          },
          1 - self.edge_model.output[2],
      }
      
      self.edge_model:forward(self.edgeInput[{ {},{4,6},{},{} }])
      local tar_edges           = {
          { 
            1 - self.edge_model.output[1][1] ,
          1 - self.edge_model.output[1][2],
          1 - self.edge_model.output[1][3],
          1 - self.edge_model.output[1][4],
          1 - self.edge_model.output[1][5], 
          },
          1 - self.edge_model.output[2],
      }
      
      -- begin train actual USOF
      -- local output              = self.model:forward{self.input, ref_edges[1]}:float()
      local output              = self.model:forward{ self.input , ref_edges[1]}:float()
      local batchSize           = output:size(1)
      
      local estimated_refImg    = self.warp_image:forward{self.input[{ {},{4,6},{},{} }] , self.model.output}
      local estimated_refEdge   = self.warp_edge:forward{tar_edges[2]:cuda() , self.model.output}


      local photo_loss          = self.criterion:forward(self.warp_image.output , self.input[{ {},{1,3},{},{} }])
      local smooth_loss         = self.smoothness_loss:forward(self.model.output , nil)*self.opt.smooth_weight
      local edge_loss           = self.edge_loss:forward(self.warp_edge.output, ref_edges[2])*self.opt.edge_weight
      
      criterion_output          = photo_loss + smooth_loss + edge_loss
      debug_loss                = debug_loss + (photo_loss + smooth_loss + edge_loss)

      -- criterion_output = photo_loss
      -- debug_loss = debug_loss + photo_loss

      local total_gradInput
      self.model:zeroGradParameters()

    --    self.criterion:backward(self.warp_image.output , MSECriterion_target)
    --   local photo_grads     = self.warp_image:backward( { self.input[{ {},{1,3},{},{} }] , {self.input[{ {},{4,6},{},{} }] , self.model.output} } , self.criterion.gradInput)

      self.criterion:backward(self.warp_image.output , self.input[{ {},{1,3},{},{} }])
      self.edge_loss:backward(self.warp_edge.output, ref_edges[2])
      
      local photo_grads     = self.warp_image:backward({self.input[{ {},{4,6},{},{} }] , self.model.output} , self.criterion.gradInput)
      local smooth_grads    = self.smoothness_loss:backward(self.model.output , nil)*self.opt.smooth_weight
      local edge_grads      = self.warp_edge:backward({tar_edges[2]:cuda() , self.model.output} , (self.opt.edge_weight*self.edge_loss.gradInput))

      total_gradInput       = (photo_grads[2]:transpose(3,4):transpose(2,3) + smooth_grads + edge_grads[2]:transpose(3,4):transpose(2,3))
      -- total_gradInput       = (photo_grads[2]:transpose(3,4):transpose(2,3))


      self.model:backward({self.input, ref_edges[1]} , total_gradInput)
      -- self.model:backward({torch.cat(torch.cat(self.input, ref_edges[2], 2), tar_edges[2], 2), ref_edges[1]} , total_gradInput)


      local _, tmp_loss = optim.adam(feval, self.params, self.optimState)
      -- local _, tmp_loss = optim.sgd(feval, self.params, self.optimState)


      lossSum = lossSum + criterion_output
	
      N = n

      if (n%100) == 0 then
     
          print(string.format('Gradient min: %1.4f \t max:  %1.4f \t norm: %1.4f', torch.min(self.gradParams:float()), torch.max(self.gradParams:float()), torch.norm(self.gradParams:float())))
          print(string.format('output u_min: %1.4f \t u_max:%1.4f',torch.min(output[{ {1},{1},{},{} }]),torch.max(output[{ {1},{1},{},{} }])))
          print(string.format('output v_min: %1.4f \t v_max:%1.4f',torch.min(output[{ {1},{2},{},{} }]),torch.max(output[{ {1},{2},{},{} }])))

          image.save('losses/current_warped_img.png', estimated_refImg[{ {1},{},{},{} }]:reshape(3,estimated_refImg:size(3),estimated_refImg:size(4)))
          image.save('losses/current_warped_edge.png', estimated_refEdge[{ {1},{},{},{} }]:reshape(1,estimated_refEdge:size(3),estimated_refEdge:size(4)))

          local tmpOut = torch.zeros(1, 2, output:size(3),output:size(4))
          tmpOut[{ {1},{1},{},{} }] = output[{ {1},{2},{},{} }]:clone()
          tmpOut[{ {1},{2},{},{} }] = output[{ {1},{1},{},{} }]:clone()
          local tmpIMG = uvToColor(tmpOut[{ {1},{},{},{} }]:reshape(2,output:size(3),output:size(4))):div(255)
          local gtIMG  = uvToColor(self.gt_flow[{ {1},{1,2},{},{} }]:float():reshape(2, self.gt_flow:size(3), self.gt_flow:size(4))):div(255)
          image.save('losses/gt_flow.png', gtIMG)
    	    image.save('losses/current_flow.png', tmpIMG)
          image.save('losses/current_ref.png', self.input[{ {1},{1,3},{},{} }]:reshape(3,self.input[{ {1},{1,3},{},{} }]:size(3),self.input[{ {1},{1,3},{},{} }]:size(4)))
          image.save('losses/current_tar.png', self.input[{ {1},{4,6},{},{} }]:reshape(3,self.input[{ {1},{4,6},{},{} }]:size(3),self.input[{ {1},{4,6},{},{} }]:size(4)))

          image.save('losses/current_refEdge.png', ref_edges[2][{ {1},{},{},{} }]:reshape(1,ref_edges[2][{ {1},{},{},{} }]:size(3),ref_edges[2][{ {1},{},{},{} }]:size(4)))
          image.save('losses/current_tarEdge.png', tar_edges[2][{ {1},{},{},{} }]:reshape(1,tar_edges[2][{ {1},{},{},{} }]:size(3),tar_edges[2][{ {1},{},{},{} }]:size(4)))

      end
	if (n%10) == 0 then
   	     print((' | Epoch: [%d][%d/%d]    Time %.3f  Data %.3f  loss %1.4f'):format(
             epoch, n, trainSize, timer:time().real, dataTime, criterion_output))--total_loss))
   	   -- check that the storage didn't get changed due to an unfortunate getParameters call
   	end

 	    assert(self.params:storage() == self.model:parameters()[1]:storage())

      timer:reset()
      dataTimer:reset()
   end

--    return top1Sum / N, top5Sum / N, lossSum / N
    return lossSum / N
end

function Trainer:test(epoch, dataloader)

   local timer = torch.Timer()
   local size = dataloader:size()
   local avgEPE,errPixels   = 0.0, 0.0
   local N                  = 0
   local criterion_output   = 0.0

   self.model:evaluate()
   for n, sample in dataloader:run() do
      -- Copy input and target to the GPU
      self:copyInputs(sample)

      -- run edge detector
      self.edge_model:forward(self.edgeInput[{ {},{1,3},{},{} }])
      local ref_edges           = {
          { 
            1 - self.edge_model.output[1][1] ,
          1 - self.edge_model.output[1][2],
          1 - self.edge_model.output[1][3],
          1 - self.edge_model.output[1][4],
          1 - self.edge_model.output[1][5], 
          },
          1 - self.edge_model.output[2],
      }
      
      self.edge_model:forward(self.edgeInput[{ {},{4,6},{},{} }])
      local tar_edges           = {
          { 
            1 - self.edge_model.output[1][1] ,
          1 - self.edge_model.output[1][2],
          1 - self.edge_model.output[1][3],
          1 - self.edge_model.output[1][4],
          1 - self.edge_model.output[1][5], 
          },
          1 - self.edge_model.output[2],
      }
      


      -- begin train actual USOF
      -- local output              = self.model:forward{self.input, ref_edges[1]}:float()
      local output              = self.model:forward{ self.input , ref_edges[1]}:float()
      local batchSize           = output:size(1)
      
      local estimated_refImg    = self.warp_image:forward{self.input[{ {},{4,6},{},{} }] , self.model.output}
      local estimated_refEdge   = self.warp_edge:forward{tar_edges[2] , self.model.output}

      local photo_loss          = self.criterion:forward(self.warp_image.output , self.input[{ {},{1,3},{},{} }])
      local smooth_loss         = self.smoothness_loss:forward(self.model.output , nil)*self.opt.smooth_weight
      local edge_loss           = self.edge_loss:forward(self.warp_edge.output, ref_edges[2])*self.opt.edge_weight

      criterion_output = photo_loss + smooth_loss + edge_loss

      N = n

   	  local average_epe, erroneous_pixels = evaluateEPE(self.model.output, self.gt_flow, 3)

      if (n%50 == 0) then    
           
          
          for ii = 1, 2 do
            local tmpOut = torch.zeros(1, 2, output:size(3),output:size(4))
            tmpOut[{ {1},{1},{},{} }] = output[{ {ii},{2},{},{} }]:clone()
            tmpOut[{ {1},{2},{},{} }] = output[{ {ii},{1},{},{} }]:clone()
            local tmpIMG = uvToColor(tmpOut:reshape(2,output:size(3),output:size(4))):div(255)
            image.save('losses/testing/est_flow_' .. ii .. '.png', tmpIMG)
            local gtIMG  = uvToColor(self.gt_flow[{ {ii},{1,2},{},{} }]:float():reshape(2, self.gt_flow:size(3), self.gt_flow:size(4))):div(255)
            image.save('losses/testing/gt_flow_' .. ii .. '.png', gtIMG)

            image.save('losses/testing/test_ref_' .. ii .. '.png', self.input[{ {ii},{1,3},{},{} }]:reshape(3,self.input[{ {ii},{1,3},{},{} }]:size(3),self.input[{ {ii},{1,3},{},{} }]:size(4)))
            image.save('losses/testing/test_tar_' .. ii .. '.png', self.input[{ {ii},{4,6},{},{} }]:reshape(3,self.input[{ {ii},{4,6},{},{} }]:size(3),self.input[{ {ii},{4,6},{},{} }]:size(4)))
            image.save('losses/testing/current_warped_img_' .. ii .. '.png', estimated_refImg[{ {1},{},{},{} }]:reshape(3,estimated_refImg:size(3),estimated_refImg:size(4)))

            image.save('losses/testing/test_refEdge_' .. ii .. '.png', ref_edges[2][{ {ii},{},{},{} }]:reshape(1,ref_edges[2][{ {ii},{},{},{} }]:size(3),ref_edges[2][{ {ii},{},{},{} }]:size(4)))
            image.save('losses/testing/test_tarEdge_' .. ii .. '.png', tar_edges[2][{ {ii},{},{},{} }]:reshape(1,tar_edges[2][{ {ii},{},{},{} }]:size(3),tar_edges[2][{ {ii},{},{},{} }]:size(4)))
            image.save('losses/testing/current_warped_edge_' .. ii .. '.png', estimated_refEdge[{ {ii},{},{},{} }]:reshape(1,estimated_refEdge:size(3),estimated_refEdge:size(4)))          
          end
          collectgarbage()
      end
          print((' | Test: [%d][%d/%d]    Time %.3f  loss %1.4f Average EPE %1.3f erroneous_pixels %1.3f'):format( epoch, n, size, timer:time().real, criterion_output, average_epe, erroneous_pixels))
          -- print((' | Test: [%d][%d/%d]    Time %.3f  '):format( epoch, n, size, timer:time().real, criterion_output))

      avgEPE = avgEPE + average_epe
      errPixels= errPixels + erroneous_pixels

      timer:reset()
   end
   self.model:training()

  --  return EPE_below_total/N, EPE_above_total/N, EPE_all_total/N, criterion_output/N
   return criterion_output/N, avgEPE/N, errPixels/N
end

function Trainer:copyInputs(sample)
   -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
   -- if using DataParallelTable. The target is always copied to a CUDA tensor
   self.input = self.input or (self.opt.nGPU == 1
      and torch.CudaTensor()
      or cutorch.createCudaHostTensor())
   self.gt_flow = self.gt_flow or torch.CudaTensor()

   self.input:resize(sample.input:size()):copy(sample.input)
   self.gt_flow:resize(sample.gt_flow:size()):copy(sample.gt_flow)

    local image_for_edge             = torch.CudaTensor(self.input:size())
    image_for_edge[{ {},{1},{},{} }] = self.input[{ {},{3},{},{} }]:clone()
    image_for_edge[{ {},{2},{},{} }] = self.input[{ {},{2},{},{} }]:clone()
    image_for_edge[{ {},{3},{},{} }] = self.input[{ {},{1},{},{} }]:clone()
    image_for_edge[{ {},{4},{},{} }] = self.input[{ {},{6},{},{} }]:clone()
    image_for_edge[{ {},{5},{},{} }] = self.input[{ {},{5},{},{} }]:clone()
    image_for_edge[{ {},{6},{},{} }] = self.input[{ {},{4},{},{} }]:clone()
      
    image_for_edge:mul(255):floor()
    image_for_edge[{ {},{1},{},{} }]:csub(104.00698793)
    image_for_edge[{ {},{2},{},{} }]:csub(116.66876762)
    image_for_edge[{ {},{3},{},{} }]:csub(122.67891434)
    image_for_edge[{ {},{4},{},{} }]:csub(104.00698793)
    image_for_edge[{ {},{5},{},{} }]:csub(116.66876762)
    image_for_edge[{ {},{6},{},{} }]:csub(122.67891434)

    self.edgeInput = image_for_edge

end

function Trainer:learningRate(epoch)
   -- Training schedule
   if (self.opt.dataset == 'FlyingChairs') and (epoch >= 15) then
      if (epoch%15 == 0) then
  --  if (self.opt.dataset == 'FlyingChairs') and (epoch >= 20) then
  --     if (epoch%20 == 0) then
      	return self.optimState.learningRate/10
      else
        return self.optimState.learningRate
      end
   else
	    return self.optimState.learningRate
   end 
end

return M.Trainer
