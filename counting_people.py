import tensorflow as tf
import tiny_face_model
import cv2
import util
import glob
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import pickle
import traceback
import pylab as pl
import time
import os
import os.path
import sys
import pickle
import pandas as pd
import h5py

from argparse import ArgumentParser
from keras.models import load_model
from scipy import misc
from scipy.misc import imread, imresize
from scipy.special import expit
from scipy.spatial import distance
from keras.models import load_model
from keras.utils import plot_model
from keras.models import Model

MAX_INPUT_DIM = 5000.0

def get_distance_points(refined_bboxes, refined_bboxes_anterior): 
  
  something = []
  for r in refined_bboxes_anterior:
    _r = [int(x) for x in r[:4]]
    distancia = 15.0 #limita espaço de procura
    #0 -> x1, 1 -> y1 , 2 -> x2, 3 -> y2
    c1 = ((_r[0] + _r[2])/2), ((_r[1] + _r[3])/2)
    c2 = c1
    for row in refined_bboxes:
      _row = [int(x) for x in row[:4]]
      c_rect2 = ((_row[0] + _row[2])/2), ((_row[1] + _row[3])/2)
      if(distance.euclidean(c1, c_rect2)<distancia):
        distancia = distance.euclidean(c1, c_rect2)
        c2 = ((_row[0] + _row[2])/2), ((_row[1] + _row[3])/2)
    if(c1!=c2):    
      something.append([c1, c2])
  
  return something

def draw_distance(raw_image, distancias):
  #write in image the distances
  for something in distancias[-12:]: #utilize 12 últimos frames
    cont = 0
    tam = len(something)
    while(cont<tam):
      p1 = something[cont][0] #ponto atual
      p2 = something[cont][1] #ponto antigo
      cv2.line(raw_image, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 255), 1)
      cont = cont + 1 

def draw_contador(raw_image, contador):
  cv2.putText(raw_image, str(contador), (280, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


def overlay_bounding_boxes(raw_img, refined_bboxes, lw, num):
  """Overlay bounding boxes of face on images.
    Args:
      raw_img:
        A target image.
      refined_bboxes:
        Bounding boxes of detected faces.
      lw: 
        Line width of bounding boxes. If zero specified,
        this is determined based on confidence of each detection.
    Returns:
      None.
  """
  # Overlay bounding boxes on an image with the color based on the confidence.
  
  for r in refined_bboxes:
    _score = expit(r[4])
    cm_idx = int(np.ceil(_score * 255))
    rect_color = [int(np.ceil(x * 255)) for x in util.cm_data[cm_idx]]  # parula
    _lw = lw
    if lw == 0:  # line width of each bounding box is adaptively determined.
      bw, bh = r[2] - r[0] + 1, r[3] - r[0] + 1
      _lw = 1 if min(bw, bh) <= 20 else max(2, min(3, min(bh / 20, bw / 20)))
      _lw = int(np.ceil(_lw * _score))
    _r = [int(x) for x in r[:4]]
    #0 -> x1, 1 -> y1 , 2 -> x2, 3 -> y2
    # x - horizontal e y - vertical
    #face = raw_img[int(r[1]):int(_r[3]), int(_r[0]):int(_r[2])]
    
    #name = 'A:\\TCC\\Tiny_Faces_in_Tensorflow\\v3\\faces2\\face'+str(num)+'_'+str(cont)+'.jpg'
    #face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
    #cv2.imwrite(name, face)
    cv2.rectangle(raw_img, (_r[0], _r[1]), (_r[2], _r[3]), (255, 255, 0), int(_lw/2))
 
def get_faces_distances(refined_bboxes, faces, n):
  #euma matriz de faces. cada coluna é um frame, salva as faces proximas
  #if refined_bboxes_anterior equals []  não faça nada 
  #vetor com faces achadas - cada coluna 
  distancia = 15.0  #limita espaço de procura
  zero = (0.0 , 0.0)
  nao_foi = True


  for row in refined_bboxes:
    _row = [int(x) for x in row[:4]]
    c1 = ((_row[0] + _row[2])/2), ((_row[1] + _row[3])/2)
    if(n==0):
      faces.append([c1])      
    else:
      for face in faces:  
        if(face[n-1]!=zero):
          if(distance.euclidean(c1, face[n-1])<distancia):
            if(len(face)<(n+1)):
              face.append(c1)
              nao_foi = False
            break
        elif(face[n-2]!=zero):
          if(distance.euclidean(c1, face[n-2])<distancia):
            if(len(face)<(n+1)):
              face.append(c1)
              nao_foi = False
            break
        elif(face[n-3]!=zero):
          if(distance.euclidean(c1, face[n-3])<distancia):
            if(len(face)<(n+1)):
              face.append(c1)
              nao_foi = False
            break
        elif(face[n-4]!=zero):
          if(distance.euclidean(c1, face[n-4])<distancia):
            if(len(face)<(n+1)):
              face.append(c1)
              nao_foi = False
            break
        elif(face[n-5]!=zero):
          if(distance.euclidean(c1, face[n-5])<distancia):
            if(len(face)<(n+1)):
              face.append(c1)
              nao_foi = False
            break
      if(nao_foi):
        nova_face = []
        for x in range(0, n-1):
          nova_face.append(zero)
        nova_face.append(c1)
        nao_foi=True
        faces.append(nova_face)
       
      
  for face in faces:
    if(len(face)<(n+1)):
      face.append(zero)

  
  #caso tenha medir a diferença de um ponto a sua escolha


def draw_distance_labels_counter(faces, raw_image):
  zero = (0.0 , 0.0)
  labels = 0
  for face in faces:
    if(face[-1:][0]!=zero):
      ponto = face[-1:][0]
      cv2.putText(raw_image, str(labels), (int(ponto[0]), int(ponto[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    elif(face[-2:][0]!=zero):
      ponto = face[-2:][0]
      cv2.putText(raw_image, str(labels), (int(ponto[0]), int(ponto[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    frames = face[-12:] #utilize 12 últimos frames  
    cont = 1
    tam = len(frames)
    while(cont<tam):
      p1 = frames[cont] #ponto atual
      p2 = frames[cont-1] #ponto antigo
      if((p1!=zero) == (p2!=zero)):
        cv2.line(raw_image, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 255), 1)
      cont = cont + 1
    labels = labels + 1


def evaluate(weight_file_path, data_dir, output_dir, fps, prob_thresh=0.5, nms_thresh=0.1, lw=3, display=True):
  """Detect faces in images.
  Args:
    prob_thresh:
        The threshold of detection confidence.
    nms_thresh:
        The overlap threshold of non maximum suppression
    weight_file_path: 
        A pretrained weight file in the pickle format 
        generated by matconvnet_hr101_to_tf.py.
    data_dir: 
        A directory which contains images.
    output_dir: 
        A directory into which images with detected faces are output.
    lw: 
        Line width of bounding boxes. If zero specified,
        this is determined based on confidence of each detection.
    display:
        Display tiny face images on window.
  Returns:
    None.
  """

  # placeholder of input images. Currently batch size of one is supported.
  x = tf.placeholder(tf.float32, [1, None, None, 3]) # n, h, w, c

  # Create the tiny face model which weights are loaded from a pretrained model.
  model = tiny_face_model.Model(weight_file_path)
  score_final = model.tiny_face(x)

  # Find image files in data_dir.
  filenames = []
  for ext in ('*.png', '*.gif', '*.jpg', '*.jpeg'):
    filenames.extend(glob.glob(os.path.join(data_dir, ext)))

  #Load the video 
  video = cv2.VideoCapture(data_dir)

  #buffer for traking faces
  distancias = []
  refined_bboxes_anterior = []
  faces = []
  num = 0

  #write video 
  frame_width = 352
  frame_height = 240
 
  # Define the codec and create VideoWriter object.The output is stored in 'output.avi' file.
  video_out = cv2.VideoWriter(output_dir ,cv2.VideoWriter_fourcc(*'XVID'), fps , (frame_width,frame_height))

  # Load an average image and clusters(reference boxes of templates).
  with open(weight_file_path, "rb") as f:
    _, mat_params_dict = pickle.load(f)

  average_image = model.get_data_by_key("average_image")
  clusters = model.get_data_by_key("clusters")
  clusters_h = clusters[:, 3] - clusters[:, 1] + 1
  clusters_w = clusters[:, 2] - clusters[:, 0] + 1
  normal_idx = np.where(clusters[:, 4] == 1)
  n_frame = 0

  # main
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    try:

      while (video.isOpened()):
        _, frame = video.read()
        raw_img = frame
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        raw_img_f = raw_img.astype(np.float32)
        
        def _calc_scales():
          raw_h, raw_w = raw_img.shape[0], raw_img.shape[1]
          min_scale = min(np.floor(np.log2(np.max(clusters_w[normal_idx] / raw_w))),
                          np.floor(np.log2(np.max(clusters_h[normal_idx] / raw_h))))
          max_scale = min(1.0, -np.log2(max(raw_h, raw_w) / MAX_INPUT_DIM))
          scales_down = pl.frange(min_scale, 0, 1.)
          scales_up = pl.frange(0.5, max_scale, 0.5)
          scales_pow = np.hstack((scales_down, scales_up))
          scales = np.power(2.0, scales_pow)
          return scales

        scales = _calc_scales()
        start = time.time()

        # initialize output
        bboxes = np.empty(shape=(0, 5))

        # process input at different scales
        for s in scales:
          
          img = cv2.resize(raw_img_f, (0, 0), fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
          img = img - average_image
          img = img[np.newaxis, :]

          # we don't run every template on every scale ids of templates to ignore
          tids = list(range(4, 12)) + ([] if s <= 1.0 else list(range(18, 25)))
          ignoredTids = list(set(range(0, clusters.shape[0])) - set(tids))

          # run through the net
          score_final_tf = sess.run(score_final, feed_dict={x: img})

          # collect scores
          score_cls_tf, score_reg_tf = score_final_tf[:, :, :, :25], score_final_tf[:, :, :, 25:125]
          prob_cls_tf = expit(score_cls_tf)
          prob_cls_tf[0, :, :, ignoredTids] = 0.0

          def _calc_bounding_boxes():
            # threshold for detection
            _, fy, fx, fc = np.where(prob_cls_tf > prob_thresh)

            # interpret heatmap into bounding boxes
            cy = fy * 8 - 1
            cx = fx * 8 - 1
            ch = clusters[fc, 3] - clusters[fc, 1] + 1
            cw = clusters[fc, 2] - clusters[fc, 0] + 1

            # extract bounding box refinement
            Nt = clusters.shape[0]
            tx = score_reg_tf[0, :, :, 0:Nt]
            ty = score_reg_tf[0, :, :, Nt:2*Nt]
            tw = score_reg_tf[0, :, :, 2*Nt:3*Nt]
            th = score_reg_tf[0, :, :, 3*Nt:4*Nt]

            # refine bounding boxes
            dcx = cw * tx[fy, fx, fc]
            dcy = ch * ty[fy, fx, fc]
            rcx = cx + dcx
            rcy = cy + dcy
            rcw = cw * np.exp(tw[fy, fx, fc])
            rch = ch * np.exp(th[fy, fx, fc])

            scores = score_cls_tf[0, fy, fx, fc]
            tmp_bboxes = np.vstack((rcx - rcw / 2, rcy - rch / 2, rcx + rcw / 2, rcy + rch / 2))
            tmp_bboxes = np.vstack((tmp_bboxes / s, scores))
            tmp_bboxes = tmp_bboxes.transpose()
            return tmp_bboxes

          tmp_bboxes = _calc_bounding_boxes()
          bboxes = np.vstack((bboxes, tmp_bboxes)) # <class 'tuple'>: (5265, 5)

        # non maximum suppression
        # refind_idx = util.nms(bboxes, nms_thresh)
        
        refind_idx = tf.image.non_max_suppression(tf.convert_to_tensor(bboxes[:, :4], dtype=tf.float32),
                                                    tf.convert_to_tensor(bboxes[:, 4], dtype=tf.float32),
                                                    max_output_size=bboxes.shape[0], iou_threshold=nms_thresh)
        refind_idx = sess.run(refind_idx)
        refined_bboxes = bboxes[refind_idx]
        overlay_bounding_boxes(raw_img, refined_bboxes, lw, num)

        num = num + 1

        #calcula a distância entre faces no frame atual e o frame anterior
        #retorna uma matriz com duas colunas - o centroid 1 e o centroid 2
        #dois pontos a distância entre esses pontos foi a movimentação da pessoa 
        something = get_distance_points(refined_bboxes, refined_bboxes_anterior) 
        

        get_faces_distances(refined_bboxes, faces, n_frame)
        
        draw_distance_labels_counter(faces, raw_img)
        #junta as distância com um vetor de todas as distâncias já cálculadas
        #distancias.append(something)

        
        
        #desenha a distancia entre as faces
        #a partir dos centroids das faces encotradas anteriormente
        #draw_distance(raw_img, distancias)

        

        #o frame atual se torna o anterior
        refined_bboxes_anterior = refined_bboxes
        
        # save image with bounding boxes
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)
        video_out.write(raw_img)
        n_frame = n_frame + 1

        try:
          print("time {:.2f} secs for {}_{}".format(time.time() - start, 'frame', n_frame))
        except Exception:
          traceback.print_exc()

    except Exception:
      video.release()
      video_out.release()
      traceback.print_exc()

  video.release()
  video_out.release()
  
def main():

  argparse = ArgumentParser()
  argparse.add_argument('--weight_file_path', type=str, help='Pretrained weight file.', default="/path/to/mat2tf.pkl")
  argparse.add_argument('--videos_dir', type=str, help='Video path.', default="/path/to/video")
  argparse.add_argument('--videos_output_dir', type=str, help='Output Video path with faces detected.', default="/path/to/video_output")
  argparse.add_argument('--prob_thresh', type=float, help='The threshold of detection confidence(default: 0.5).', default=0.5)
  argparse.add_argument('--nms_thresh', type=float, help='The overlap threshold of non maximum suppression(default: 0.1).', default=0.1)
  argparse.add_argument('--line_width', type=int, help='Line width of bounding boxes(0: auto).', default=3)
  argparse.add_argument('--display', type=bool, help='Display each image on window.', default=True)
  argparse.add_argument('--fps', type=int, help='Frames por segundo', default=30)

  args = argparse.parse_args()

  # check arguments
  assert os.path.exists(args.weight_file_path), "weight file: " + args.weight_file_path + " not found."
  assert os.path.exists(args.videos_dir), "data directory: " + args.video_dir + " not found."

  assert args.line_width >= 0, "line_width should be >= 0."

  with tf.Graph().as_default():
    evaluate(
      weight_file_path=args.weight_file_path, data_dir=args.videos_dir, output_dir=args.videos_output_dir,
      prob_thresh=args.prob_thresh, nms_thresh=args.nms_thresh,
      lw=args.line_width, display=args.display, fps=args.fps)

if __name__ == '__main__':
  main()
