from init import *


def TrustDetect(video, conf_thresh, iou_thresh, trust_acc_thresh, trust_iou_thresh, debug, save_path = None):

    print(f'\nvideo = {video}, conf_thresh = {conf_thresh}, iou_thresh = {iou_thresh}, trust_acc_thresh = {trust_acc_thresh}, trust_iou_thresh = {trust_iou_thresh}, debug = {debug}, save_path = {save_path}')

    video = cv2.VideoCapture(video)

    video_fps = video.get(cv2.CAP_PROP_FPS)
    video_frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)

    frame_no = 0

    current_fps = video_fps

    trust_fps = 1

    trust_frame = int(max(1, video_fps // trust_fps))

    print('Video_FPS :', video_fps)
    print('trust_frame :', trust_frame)

    acc = None
    mean_iou = None

    detect_status = 'ok'

    msg = None
    prev_msg = None

    fps = FPS().start() 

    writer = None

    while True:

        _, frame = video.read()
        
        if frame is None:

            break

        frame_processed = frame.copy()
        rgb = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2RGB)


        if frame_no % trust_frame != 0:

            if detect_status == 'ok':
                dets = detect(rgb, model=model_sm, conf_thresh=conf_thresh, iou_thresh=iou_thresh)
            else:
                dets = detect(rgb, model=model_lrg, conf_thresh=conf_thresh, iou_thresh=iou_thresh)

            for det in dets:
                x1, y1, x2, y2, conf, cls = det.cpu().detach().numpy()
                startX, startY, endX, endY = int(x1), int(y1), int(x2), int(y2)

                cv2.rectangle(frame_processed, (startX, startY), (endX, endY), get_color(int(cls)+2), 2)
                draw_bb_text(frame_processed, class_names[int(cls)], (startX, startY, endX, endY),cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 0), 1,get_color(int(cls)+2))


        elif frame_no % trust_frame == 0:

            eta = (video_frame_count - frame_no) / current_fps

            eta_str = "%.2d"%(eta // 3600)+":%.2d"%((eta // 60) % 60)+":%.2d"%(eta % 60)

            print(f'\nProcessing Frame : {frame_no} ({round(((frame_no+1) * 100) / video_frame_count,2)} %), Current FPS : {current_fps}, ETA : {eta_str}')

            fps = FPS().start() 

            
            dets_lrg = detect(rgb, model=model_lrg, conf_thresh=conf_thresh, iou_thresh=iou_thresh)
            dets_sm = detect(rgb, model=model_sm, conf_thresh=conf_thresh, iou_thresh=iou_thresh)

            
                

            # print(f'length lrg : {len(dets_lrg)}, sm : {len(dets_sm)}')

            # try:

            all_ious = []

            tp, fp, fn = 0, 0, 0

            for i,det_lrg in enumerate(dets_lrg):
                ious = [get_iou(det_lrg[:4],det_sm[:4]) for det_sm in dets_sm]
                j = np.argmax(ious)
                if i == np.argmax([get_iou(det_lrg[:4],dets_sm[j][:4]) for det_lrg in dets_lrg]) and dets_lrg[i][-1] == dets_sm[j][-1] and get_iou(dets_lrg[i][:4],dets_sm[j][:4]) > trust_iou_thresh:
                    all_ious.append(float(max(ious)))
                    det = dets_lrg[i]
                    x1, y1, x2, y2, conf, cls = det.cpu().detach().numpy()
                    startX, startY, endX, endY = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(frame_processed, (startX, startY), (endX, endY), (100, 255, 100), 2)
                    draw_bb_text(frame_processed, class_names[int(cls)], (startX, startY, endX, endY),cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 0), 1, (100, 255, 100))
                    tp += 1
                else:
                    all_ious.append(0)
                    det = dets_lrg[i]
                    x1, y1, x2, y2, conf, cls = det.cpu().detach().numpy()
                    startX, startY, endX, endY = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(frame_processed, (startX, startY), (endX, endY), (255, 100, 100), 2)
                    draw_bb_text(frame_processed, class_names[int(cls)], (startX, startY, endX, endY),cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 0), 1, (255, 100, 100))
                    fn += 1

            

            for i,det_sm in enumerate(dets_sm):
                ious = [get_iou(det_sm[:4],det_lrg[:4]) for det_lrg in dets_lrg]
                j = np.argmax(ious)
                if i == np.argmax([get_iou(det_sm[:4],dets_lrg[j][:4]) for det_sm in dets_sm]) and dets_sm[i][-1] == dets_lrg[j][-1] and get_iou(dets_sm[i][:4],dets_lrg[j][:4]) > trust_iou_thresh:
                    pass
                else:
                    all_ious.append(0)
                    det = dets_sm[i]
                    x1, y1, x2, y2, conf, cls = det.cpu().detach().numpy()
                    startX, startY, endX, endY = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(frame_processed, (startX, startY), (endX, endY), (100, 100, 255), 2)
                    draw_bb_text(frame_processed, class_names[int(cls)], (startX, startY, endX, endY),cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 0), 1,(100, 100, 255))
                    fp += 1

            acc = tp / (tp+fp+fn)

            all_ious = list(set(all_ious))
            mean_iou = np.mean(all_ious)

            print(f'TP : {tp}, FP : {fp}, FN : {fn}, MIOU : {round(mean_iou,2)}, ACC : {round(acc, 2)}')

            if acc > trust_acc_thresh:
                detect_status = 'ok'
            else:
                detect_status = 'not_ok'



            # print(all_ious)
            # print('all_ious :', len(all_ious))
            # print('all_ious :', len(all_ious))
        

            # except:
                # print('except : ',mean_iou)
                # mean_iou = mean_iou

            
            
        if detect_status == 'ok':
            msg = 'Small Model is doing good!'
            bg_color = (255, 100, 100)
        else:
            msg = 'Detected error in small model! Now, inferencing large model.'
            bg_color = (100, 100, 255)

        

        draw_text(frame_processed, f"Model : {'small' if detect_status == 'ok' else 'large'}", (2,20),cv2.FONT_HERSHEY_DUPLEX, 0.6, (50, 50, 50), 2, (200, 200, 200))
        draw_text(frame_processed, f"MIOU : {round(mean_iou, 2) if mean_iou is not None else 'None'}" , (2,40),cv2.FONT_HERSHEY_DUPLEX, 0.6, (50, 50, 50), 2, (200, 200, 200))
        draw_text(frame_processed, f"ACC : {round(acc, 2) if acc is not None else 'None'}", (2,60),cv2.FONT_HERSHEY_DUPLEX, 0.6, (50, 50, 50), 2, (200, 200, 200))
        draw_text(frame_processed, f"FPS : {round(current_fps, 2) if current_fps is not None else 'None'}", (2,80),cv2.FONT_HERSHEY_DUPLEX, 0.6, (50, 50, 50), 2, (200, 200, 200))
        draw_text(frame_processed, msg, (2,100),cv2.FONT_HERSHEY_DUPLEX, 0.6, (50, 50, 50), 2, bg_color)
        
        
        
        if msg != prev_msg:
            time_stamp = frame_no // video_fps

            time_stamp_str = "%.2d"%(time_stamp // 3600)+":%.2d"%((time_stamp // 60) % 60)+":%.2d"%(time_stamp % 60)
            if msg is not None:
                print({"time":time_stamp_str, "mean_iou": mean_iou, "message" : msg })

            prev_msg = msg

        
        
        
        
        
        if debug:

            cv2.imshow('debug display - trust object detection', frame_processed)

            key = cv2.waitKey(1)

            if key == ord('q'):
              break


        fps.update()
        fps.stop()

        frame_no += 1
        current_fps = fps.fps()

        if save_path is not None and writer is None:   
            fourcc = cv2.VideoWriter_fourcc('V','P','8','0')
            writer = cv2.VideoWriter(save_path, fourcc, video_fps, (frame_processed.shape[1], frame_processed.shape[0]), True)
            
        if writer is not None:
            writer.write(frame_processed)

    if writer is not None:
        writer.release()

        
    video.release()

    if debug:
        cv2.destroyWindow(f'debug display - trust object detection')




TrustDetect(video='demo_videos\ipcam1.mp4', conf_thresh = 0.45, iou_thresh = 0.4, trust_acc_thresh = 0.7, trust_iou_thresh = 0.7, 
            debug=True, save_path='out1.webm')