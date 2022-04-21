import torch
import LoadData
import TextFeature
import ImageFeature

class RepresentationFusion_1(torch.nn.Module):
    def __init__(self,att1_feature_size):
        super(RepresentationFusion_1, self).__init__()
        self.linear1_1 = torch.nn.Linear(att1_feature_size+att1_feature_size, int((att1_feature_size+att1_feature_size)/2))
        self.linear2_1 = torch.nn.Linear(int((att1_feature_size+att1_feature_size)/2), 1)
    def forward(self, feature1,feature1_seq):
        output_list_1=list()
        length=feature1_seq.size(0)
        for i in range(length):
            output1=torch.tanh(self.linear1_1(torch.cat([feature1_seq[i],feature1],dim=1)))                               #[32，1024]    [32,1024]
            output_list_1.append(self.linear2_1(output1))
        weight_1=torch.nn.functional.softmax(torch.torch.stack(output_list_1),dim=0)
        output=torch.mean((weight_1)*feature1_seq,0)
        return output
		
class ModalityFusion_1(torch.nn.Module):
    def __init__(self,feature1_size):
        super(ModalityFusion_1, self).__init__()
        self.feature1_size=feature1_size               
        self.m1_attention=RepresentationFusion_1(self.feature1_size)
        self.m1_linear_1=torch.nn.Linear(self.feature1_size,512)
        self.m1_linear_2=torch.nn.Linear(512,1)
        self.m1_linear_3=torch.nn.Linear(self.feature1_size,512)

    def forward(self, m1_feature,m1_seq):
        vector1=self.m1_attention(m1_feature,m1_seq)
        m1_hidden=torch.tanh(self.m1_linear_1(vector1))
        m1_score=self.m1_linear_2(m1_hidden)
        score=torch.nn.functional.softmax(torch.stack([m1_score]),dim=0)
        vector1=torch.tanh(self.m1_linear_3(vector1))

        # final fuse
        output=score[0]*vector1
        
        return output			
		
	
	
if __name__ == "__main__":
    image=ImageFeature.ExtractImageFeature()
    text=TextFeature.ExtractTextFeature(LoadData.TEXT_LENGTH, LoadData.TEXT_HIDDEN)
    fuse_image=ModalityFusion_1(1024)
    fuse_text=ModalityFusion_1(512)
    fuse_image_text=ModalityFusion_2(1024,512)
    for text_index,image_feature,group,id in LoadData.train_loader:
        image_result,image_seq=image(image_feature)
        text_result,text_seq=text(text_index,None)
        result_image=fuse_image(image_result,image_seq)
        #result_text=fuse_text(text_result,text_seq.permute(1,0,2))
        # result_caption=fuse_caption(caption_result,caption_seq.permute(1,0,2))
        # result_image_text=fuse_image_text(image_result,image_seq,text_result,text_seq.permute(1,0,2))
        # result_text_caption=fuse_text_caption(text_result,text_seq.permute(1,0,2),caption_result,caption_seq.permute(1,0,2))
        # result_image_caption=fuse_image_caption(image_result,image_seq,caption_result,caption_seq.permute(1,0,2))
        # result_image_text_caption=fuse_image_text_caption(image_result,image_seq,text_result,text_seq.permute(1,0,2),caption_result,caption_seq.permute(1,0,2))
        print("图片：",result_image)
        break