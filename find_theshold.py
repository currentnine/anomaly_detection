def find_theshold(model):
    model.eval()


    for batch_images in train_dataloader:   # images.shape (8배치사이즈)   
        
    
        batch_images = batch_images.permute(0,3,2,1)
        batch_images = batch_images.float()
        batch_images = batch_images.cuda()

        with torch.no_grad():
            ret = model(batch_images)  #divide_num 갯수만큼의 이미지 모델에 넣고 output출력

        
        # outputs = ret["anomaly_map"].detach()
        preds = ret['preds']

        score = torch.mean(torch.stack(preds), 0).cpu().numpy()
        good_scores.extend(score.tolist())

       
        

    threshold  = np.percentile(good_scores, 98)

    return threshold



def eval_one_image(model, threshold, path):

    img  = cv2.imread(path)
    img = img.unsqueeze(0)

    img = img.permute(0,3,2,1)
    img = img.float()
    img = img.cuda()

    with torch.no_grad():
        ret = model(img)

    pred = ret['preds']
    score = torch.mean(torch.stack(pred), 0).cpu().numpy()

    if score > threshold:
        print('bad')

    else:
        print('good')