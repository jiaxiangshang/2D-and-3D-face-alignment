--[[
nvidia-docker run -it 1adrianb/facealignment-torch
docker exec -it fan_torch /bin/bash
docker exec -it -v /data0/0_DATA:/data centos /bin/bash
--]]
string.split = function(s, p)
    local rt= {}
    string.gsub(s, '[^'..p..']+', function(w) table.insert(rt, w) end )
    return rt
end

function getGlobalList(data_path)
    print('getGlobalList...')

    local filesList = {}
    cnt = 1
    file = io.open(data_path,"r");
    for line in file:lines() do
        print(cnt ..":".. line)
        filesList[cnt] = line
        cnt = cnt + 1
    end
    file:close()

    print('Found '..#filesList..' images')
    return filesList
end

function getBBox(data_path)
    print('getBBox...')

    file = io.open(data_path,"r");
    for line in file:lines() do
        line_bbox = line
    end
    -- print(line_bbox)
    bbox = string.split(line_bbox, ',')
    file:close()
    print('Found '..#bbox..' axis')
    return bbox
end

local data_top = '/data0/0_DATA/0_Face_3D/3_ESRC_OBJ/'
local name_gl = 'train_render.txt'
local data_path = data_top..name_gl
local fileList = getGlobalList(data_path)

for i = 1, #fileList do
    print("processing ", fileList[i])

    local list_gl = string.split(fileList[i], ' ')

    local name_subfolder = list_gl[1]
    local name_pure = list_gl[2]

    local path_image = data_top..name_subfolder..'/'..name_pure..'.jpg'
    local path_bbox = data_top..name_subfolder..'/'..name_pure..'_bbox.txt'

    print(path_image, path_bbox)

    local bbox = getBBox(path_bbox)
    local minX, minY, maxX, maxY, score = unpack(bbox)

    print(bbox, minX, minY, maxX, maxY, score)
    local img = image.load(path_image)

    -- Convert grayscale to pseudo-rgb
    if img:size(1)==1 then
        print("\n-- Convert grayscale to pseudo-rgb")
        img = torch.repeatTensor(img,3,1,1)
    end
end