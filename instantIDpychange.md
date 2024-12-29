# Add this function to InstantID.py, replacing the existing extractFeatures function

def select_face(faces, method='largest'):
    """
    Select a face based on different criteria
    """
    if not faces:
        return None
        
    faces_with_metrics = []
    for face in faces:
        bbox = face['bbox']
        area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
        center_x = (bbox[0] + bbox[2])/2
        center_y = (bbox[1] + bbox[3])/2
        faces_with_metrics.append({
            'face': face,
            'area': area,
            'center_x': center_x,
            'center_y': center_y,
            'left': bbox[0],
            'right': bbox[2],
            'top': bbox[1],
            'bottom': bbox[3]
        })

    if method == 'largest':
        selected = max(faces_with_metrics, key=lambda x: x['area'])
    elif method == 'smallest':
        selected = min(faces_with_metrics, key=lambda x: x['area'])
    elif method == 'leftmost':
        selected = min(faces_with_metrics, key=lambda x: x['left'])
    elif method == 'rightmost':
        selected = max(faces_with_metrics, key=lambda x: x['right'])
    elif method == 'topmost':
        selected = min(faces_with_metrics, key=lambda x: x['top'])
    elif method == 'bottommost':
        selected = max(faces_with_metrics, key=lambda x: x['bottom'])
    elif method == 'center':
        selected = min(faces_with_metrics, 
                      key=lambda x: ((x['center_x']/640 - 0.5)**2 + 
                                   (x['center_y']/640 - 0.5)**2)**0.5)
    
    print(f"Selected face using '{method}' method:")
    print(f"- Area: {selected['area']:.2f}")
    print(f"- Position: ({selected['center_x']:.1f}, {selected['center_y']:.1f})")
    print(f"- Bounds: L={selected['left']:.1f}, R={selected['right']:.1f}, T={selected['top']:.1f}, B={selected['bottom']:.1f}")
    
    return selected['face']

def extractFeatures(insightface, image, extract_kps=False, selection_method='largest'):
    """Feature extraction with multiple selection methods"""
    face_img = tensor_to_image(image)
    out = []

    insightface.det_model.input_size = (640,640) # reset the detection size

    for i in range(face_img.shape[0]):
        for size in [(size, size) for size in range(640, 128, -64)]:
            insightface.det_model.input_size = size
            faces = insightface.get(face_img[i])
            if faces:
                selected_face = select_face(faces, selection_method)

                if extract_kps:
                    out.append(draw_kps(face_img[i], selected_face['kps']))
                else:
                    out.append(torch.from_numpy(selected_face['embedding']).unsqueeze(0))

                if 640 not in size:
                    print(f"\033[33mINFO: InsightFace detection resolution lowered to {size}.\033[0m")
                break

    if out:
        if extract_kps:
            out = torch.stack(T.ToTensor()(out), dim=0).permute([0,2,3,1])
        else:
            out = torch.stack(out, dim=0)
    else:
        out = None

    return out