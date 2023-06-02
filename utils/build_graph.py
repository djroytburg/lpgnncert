import pickle
#import dgl
import pandas as pd
import torch
import numpy as np
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
def build_graph_dgl(Users,Questions,Answers_dev,Answers_test):
    user_following_user=[]
    for u,f in zip(Users.uid,Users.followings):
        for v in f:
            user_following_user.append([u,v])
    user_follower_user=[]
    for u,f in zip(Users.uid,Users.followers):
        for v in f:
            user_follower_user.append([u,v])
    user_answer_write=[]
    for u,f in zip(Users.uid,Users.writtenAnswers):
        try:
            for v in f:
                user_answer_write.append([u,v])
        except:
            pass
    user_following_topic=[]
    for u,f in zip(Users.uid,Users.followedTopics):
        try:
            for v in f:
                user_following_topic.append([u,v])
        except:
            pass
    user_ask_question=[]
    for u,f in zip(Users.uid,Users.askedQuestions):
        try:
            for v in f:
                user_ask_question.append([u,v])
        except:
            pass
    user_follow_question=[]
    for u,f in zip(Users.uid,Users.followedQuestions):
        try:
            for v in f:
                user_follow_question.append([u,v])
        except:
            pass
    user_commment_answer=[]
    for u,f in zip(Users.uid,Users.commentedAnswers):
        try:
            for v in f:
                user_commment_answer.append([u,v])
        except:
            pass
    question_belongto_topic=[]
    for u,f in zip(Questions.questionId,Questions.questionTopics):
        try:
            for v in f:
                question_belongto_topic.append([u,v])
        except:
            pass
    question_have_answer=[]
    for u,f in zip(Questions.questionId,Questions.answers):
        try:
            for v in f:
                question_have_answer.append([u,v])
        except:
            pass
    val_pos_edge_index=torch.tensor(Answers_dev.apply(lambda x:[x['users'][0],x['answerId']],axis=1).values.tolist())
    test_pos_edge_index=torch.tensor(Answers_test.apply(lambda x:[x['users'][0],x['answerId']],axis=1).values.tolist())
    val_neg_edge_index=torch.tensor(Answers_dev.apply(lambda x:[x['users'][1],x['answerId']],axis=1).values.tolist())
    test_neg_edge_index=torch.tensor(Answers_test.apply(lambda x:[x['users'][1],x['answerId']],axis=1).values.tolist())
    data={
    ('user', 'write', 'answer') :(torch.tensor(user_answer_write)[:,0],torch.tensor(user_answer_write)[:,1]),
    ('user', 'comment', 'answer') :(torch.tensor(user_commment_answer)[:,0],torch.tensor(user_commment_answer)[:,1]),
    ('answer', 're_comment', 'user') :(torch.tensor(user_commment_answer)[:,1],torch.tensor(user_commment_answer)[:,0]),
    ('answer', 're_write', 'user') :(torch.tensor(user_answer_write)[:,1],torch.tensor(user_answer_write)[:,0]),
    ("user", "user_follower_user", "user"):(torch.tensor(user_follower_user)[:,0],torch.tensor(user_follower_user)[:,1]),
    ("user", "user_following_user", "user"):(torch.tensor(user_following_user)[:,0],torch.tensor(user_following_user)[:,1]),
    ("user", "user_following_topic", "topic"):(torch.tensor(user_following_topic)[:,0],torch.tensor(user_following_topic)[:,1]),
    ("topic", "re_user_following_topic", "user"):(torch.tensor(user_following_topic)[:,1],torch.tensor(user_following_topic)[:,0]),
    ("user", "user_following_question", "question"):(torch.tensor(user_follow_question)[:,0],torch.tensor(user_follow_question)[:,1]),
    ("question", "re_user_following_question", "user"):(torch.tensor(user_follow_question)[:,1],torch.tensor(user_follow_question)[:,0]),
    ("user", "ask", "question"):(torch.tensor(user_ask_question)[:,0],torch.tensor(user_ask_question)[:,1]),
    ("question", "re_ask", "user"):(torch.tensor(user_ask_question)[:,1],torch.tensor(user_ask_question)[:,0]),
    ("question", "belongto", "topic"):(torch.tensor(question_belongto_topic)[:,0],torch.tensor(question_belongto_topic)[:,1]),
    ("topic", "re_belongto", "question"):(torch.tensor(question_belongto_topic)[:,1],torch.tensor(question_belongto_topic)[:,0]),
    ("question", "have", "answer"):(torch.tensor(question_have_answer)[:,0],torch.tensor(question_have_answer)[:,1]),
    ("answer", "re_have", "question"):(torch.tensor(question_have_answer)[:,1],torch.tensor(question_have_answer)[:,0]),
    ('user','val_pos_edge_index','answer'):(val_pos_edge_index[:,0],val_pos_edge_index[:,1]),
    ('user','test_pos_edge_index','answer'):(test_pos_edge_index[:,0],test_pos_edge_index[:,1]),
    ('user','val_neg_edge_index','answer'):(val_neg_edge_index[:,0],val_neg_edge_index[:,1]),
    ('user','test_neg_edge_index','answer'):(test_neg_edge_index[:,0],test_neg_edge_index[:,1]),
    }
    g=dgl.heterograph(data)
    return g
def build_graph_pyg(Users,Questions,Topics,Answers_train,Answers_dev,Answers_test):
    user_following_user=[]
    for u,f in zip(Users.uid,Users.followings):
        for v in f:
            user_following_user.append([u,v])
    user_follower_user=[]
    for u,f in zip(Users.uid,Users.followers):
        for v in f:
            user_follower_user.append([u,v])
    user_answer_write=[]
    for u,f in zip(Users.uid,Users.writtenAnswers):
        try:
            for v in f:
                user_answer_write.append([u,v])
        except:
            pass
    user_following_topic=[]
    for u,f in zip(Users.uid,Users.followedTopics):
        try:
            for v in f:
                user_following_topic.append([u,v])
        except:
            pass
    user_ask_question=[]
    for u,f in zip(Users.uid,Users.askedQuestions):
        try:
            for v in f:
                user_ask_question.append([u,v])
        except:
            pass
    user_follow_question=[]
    for u,f in zip(Users.uid,Users.followedQuestions):
        try:
            for v in f:
                user_follow_question.append([u,v])
        except:
            pass
    user_commment_answer=[]
    for u,f in zip(Users.uid,Users.commentedAnswers):
        try:
            for v in f:
                user_commment_answer.append([u,v])
        except:
            pass
    question_belongto_topic=[]
    for u,f in zip(Questions.questionId,Questions.questionTopics):
        try:
            for v in f:
                question_belongto_topic.append([u,v])
        except:
            pass
    question_have_answer=[]
    for u,f in zip(Questions.questionId,Questions.answers):
        try:
            for v in f:
                question_have_answer.append([u,v])
        except:
            pass
    data = HeteroData()
    data["user"].node_id = torch.arange(len(Users))
    data["question"].node_id = torch.arange(len(Questions))
    data["answer"].node_id = torch.arange(len(Answers_train)+20000)
    data["topic"].node_id = torch.arange(len(Topics))
    # Add the node features and edge indices:
    data["user", "write", "answer"].edge_index =torch.tensor(np.array(user_answer_write).T.tolist())
    data["user", "comment", "answer"].edge_index =torch.tensor(np.array(user_commment_answer).T.tolist())
    data["user", "follower", "user"].edge_index=torch.tensor(np.array(user_follower_user).T.tolist())
    data["user", "following", "user"].edge_index=torch.tensor(np.array(user_following_user).T.tolist())
    data["user", "following", "topic"].edge_index=torch.tensor(np.array(user_following_topic).T.tolist())
    data["user", "following", "question"].edge_index=torch.tensor(np.array(user_follow_question).T.tolist())
    data["user", "ask", "question"].edge_index=torch.tensor(np.array(user_ask_question).T.tolist())
    data["question", "belongto", "topic"].edge_index=torch.tensor(np.array(question_belongto_topic).T.tolist())
    data["question", "have", "answer"].edge_index=torch.tensor(np.array(question_have_answer).T.tolist())
    data = T.ToUndirected()(data)
    return data
def build_meta_path_one(Users):
    meta_paths_one=[]
    for u,topics in zip(Users.uid,Users.followedTopics):
        try:
            meta_paths_one+=[[u,t] for t in topics]
        except:
            pass
    meta_paths_one=pd.DataFrame(meta_paths_one,columns=['uid','topic'])
    return meta_paths_one
def build_meta_path_two(Users,Questions):
    meta_paths_two=[]
    for u,questions in zip(Users.uid,Users.followedQuestions):
        try:
            for questionId in questions:
                try:
                    meta_paths_two+=[[u,t] for t in Questions[Questions['questionId']==questionId].questionTopics.values[0]]
                except:
                    pass
        except:
            pass
    return pd.DataFrame(meta_paths_two,columns=['uid','topic'])

def build_meta_path_three(Users,Questions):
    meta_paths_three=[]
    for u,questions in zip(Users.uid,Users.askedQuestions):
        try:
            for questionId in questions:
                try:
                    meta_paths_three+=[[u,t] for t in Questions[Questions['questionId']==questionId].questionTopics.values[0]]
                except:
                    pass
        except:
            pass
    return pd.DataFrame(meta_paths_three,columns=['uid','topic'])