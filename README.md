# fall2021_aai_project

1. Video를 N개의 Segment로 쪼갬. Segment에서 서로 다른 주기로 frame 윈도우를 뽑음 (frequency 다른 모션 detect)
   -> Online에서는 일정 주기로 frame을 list에 넣어서 하나의 segment에 해당하는 이 list를 queue에 저장
   
2. 각각의 Segment를 Temporal Difference Module에 입력으로 주고 나온 output을 Concat하면 3D motion feature map을 얻을 수 있음.

3. 이 3D motion feature map을 3D Conv나 2D Conv + temporal pooling을 해서 classification 함 (w3 attention module or something)
