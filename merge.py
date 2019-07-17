# -*- coding: utf-8 -*-

#n基于项目的协同过滤推荐算法实现
import random

import math
from operator import itemgetter


class CF():
    # 初始化参数
    def __init__(self, foodNo=20, recNo=7):
        # 找到相似的20个餐品，为目标用户推荐10部餐品,item
        self.item_n_sim_food = foodNo
        self.item_n_rec_food = recNo

        # 找到与目标用户兴趣相似的20个用户，为其推荐10个餐品,user
        self.user_n_sim_user = foodNo
        self.user_n_rec_food = recNo

        # 将数据集划分为训练集和测试集
        self.trainSet = {}
        self.testSet = {}

        # 用户相似度矩阵,item
        self.item_food_sim_matrix = {}
        self.item_food_popular = {}
        self.item_food_count = 0

        # 用户相似度矩阵,user
        self.user_user_sim_matrix = {}
        self.user_food_count = 0

        #最终推荐列表
        self.final = {}

        #权重
        self.userWeight = 6
        self.itemWeight = 10 - self.userWeight

        print('Similar food number = %d' % self.item_n_sim_food)
        print('Recommneded food number = %d' % self.item_n_rec_food)


    # 读文件得到“用户-餐品”数据
    def get_dataset(self, filename, pivot=0.75):
        for i,line in enumerate(self.load_file(filename)):
            user, food, rating, timestamp = line.split(',')
            if(random.random() < pivot):
            # if(i< 75000):
                self.trainSet.setdefault(user, {})
                self.trainSet[user][food] = rating
            else:
                self.testSet.setdefault(user, {})
                self.testSet[user][food] = rating
        print('Split trainingSet and testSet success!')
        print('TrainSet = %s' % len(self.trainSet))
        print('TestSet = %s' % len(self.testSet))


    # 读文件，返回文件的每一行
    def load_file(self, filename):
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                if i == 0:  # 去掉文件第一行的title
                    continue
                yield line.strip('\r\n')
        print('Load %s success!' % filename)


    # 计算餐品之间的相似度
    def item_calc_food_sim(self):
        for user, foods in self.trainSet.items():
            for food in foods:
                if food not in self.item_food_popular:
                    self.item_food_popular[food] = 0
                self.item_food_popular[food] += 1

        self.item_food_count = len(self.item_food_popular)
        print("item:Total food number = %d" % self.item_food_count)

        for user, foods in self.trainSet.items():
            for m1 in foods:
                for m2 in foods:
                    if m1 == m2:
                        continue
                    self.item_food_sim_matrix.setdefault(m1, {})
                    self.item_food_sim_matrix[m1].setdefault(m2, 0)
                    self.item_food_sim_matrix[m1][m2] += 1
        print("item:Build co-rated users matrix success!")

        # 计算餐品之间的相似性
        print("item:Calculating food similarity matrix ...")
        for m1, related_foods in self.item_food_sim_matrix.items():
            for m2, count in related_foods.items():
                # 注意0向量的处理，即某餐品的用户打分数为0
                if self.item_food_popular[m1] == 0 or self.item_food_popular[m2] == 0:
                    self.item_food_sim_matrix[m1][m2] = 0
                else:
                    self.item_food_sim_matrix[m1][m2] = count / math.sqrt(self.item_food_popular[m1] * self.item_food_popular[m2])
        print('item:Calculate food similarity matrix success!')


    # 针对目标用户U，找到K部相似的餐品，并推荐其N部餐品
    def recommendByItem(self, user):
        K = self.item_n_sim_food
        N = self.item_n_rec_food
        rank = {}
        watched_food = self.trainSet[user]

        for food, rating in watched_food.items():
            for related_food, w in sorted(self.item_food_sim_matrix[food].items(), key=itemgetter(1), reverse=True)[:K]:
                if related_food in watched_food:
                    continue
                rank.setdefault(related_food, 0)
                rank[related_food] += float(rating)
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[:N]


    # 产生推荐并通过准确率、召回率和覆盖率进行评估,merge
    def evaluateMerge(self):
        print('Evaluating start ...')
        N = self.item_n_rec_food
        # 准确率和召回率
        hit = 0
        rec_count = 0
        test_count = 0
        # 覆盖率
        all_rec_foods = set()
        #merge版本
        for i, user in enumerate(self.trainSet):
            # if i == 0:
                test_foods = self.testSet.get(user, {})
                item_rec_foods = self.recommendByItem(user)
                # print(item_rec_foods)
                user_rec_foods = self.recommendByUser(user)
                # print(user_rec_foods)
                rec_foods = self.mergeRecommendResult(item_rec_foods,user_rec_foods)
                # print(rec_foods)
                for food, w in rec_foods:
                    if food in test_foods:
                        hit += 1
                    all_rec_foods.add(food)
                rec_count += N
                test_count += len(test_foods)

        precision = hit / (1.0 * rec_count)
        recall = hit / (1.0 * test_count)
        f_mean = 2 * precision * recall / (precision + recall)

        # coverage = len(all_rec_foods) / (1.0 * self.item_food_count)
        print('merge:precisioin=%.4f\trecall=%.4f\tF-mean=%.4f' % (precision, recall, f_mean))

    # 产生推荐并通过准确率、召回率和覆盖率进行评估,user
    def evaluateUser(self):
        N = self.item_n_rec_food
        # 准确率和召回率
        hit = 0
        rec_count = 0
        test_count = 0
        # 覆盖率
        all_rec_foods = set()
        # merge版本
        for i, user in enumerate(self.trainSet):
            # if i == 0:
            test_foods = self.testSet.get(user, {})
            rec_foods = self.recommendByUser(user)
            # print(rec_foods)
            for food, w in rec_foods:
                if food in test_foods:
                    hit += 1
                all_rec_foods.add(food)
            rec_count += N
            test_count += len(test_foods)

        precision = hit / (1.0 * rec_count)
        recall = hit / (1.0 * test_count)
        f_mean = 2 * precision * recall / (precision + recall)

        print('user:precisioin=%.4f\trecall=%.4f\tF-mean=%.4f' % (precision, recall, f_mean))

    # 产生推荐并通过准确率、召回率和覆盖率进行评估,item
    def evaluateItem(self):
        N = self.item_n_rec_food
        # 准确率和召回率
        hit = 0
        rec_count = 0
        test_count = 0
        # 覆盖率
        all_rec_foods = set()
        # merge版本
        for i, user in enumerate(self.trainSet):
            # if i == 0:
            test_foods = self.testSet.get(user, {})
            rec_foods = self.recommendByItem(user)
            # print(rec_foods)
            for food, w in rec_foods:
                if food in test_foods:
                    hit += 1
                all_rec_foods.add(food)
            rec_count += N
            test_count += len(test_foods)

        precision = hit / (1.0 * rec_count)
        recall = hit / (1.0 * test_count)
        f_mean = 2 * precision * recall / (precision + recall)

        print('item:precisioin=%.4f\trecall=%.4f\tF-mean=%.4f' % (precision, recall, f_mean))

    # 计算用户之间的相似度
    def user_calc_user_sim(self):
        # 构建“餐品-用户”倒排索引
        # key = foodID, value = list of userIDs who have seen this food
        print('user:Building food-user table ...')
        food_user = {}
        for user, foods in self.trainSet.items():
            for food in foods:
                if food not in food_user:
                    food_user[food] = set()
                food_user[food].add(user)
        print('user:Build food-user table success!')

        self.user_food_count = len(food_user)
        print('user:Total food number = %d' % self.user_food_count)

        print('user:Build user co-rated foods matrix ...')
        for food, users in food_user.items():
            for u in users:
                for v in users:
                    if u == v:
                        continue
                    self.user_user_sim_matrix.setdefault(u, {})
                    self.user_user_sim_matrix[u].setdefault(v, 0)
                    self.user_user_sim_matrix[u][v] += 1
        print('user:Build user co-rated foods matrix success!')

        # 计算相似性t
        print('user:Calculating user similarity matrix ...')
        for u, related_users in self.user_user_sim_matrix.items():
            for v, count in related_users.items():
                self.user_user_sim_matrix[u][v] = count / math.sqrt(len(self.trainSet[u]) * len(self.trainSet[v]))
        print('user:Calculate user similarity matrix success!')

    # 针对目标用户U，找到其最相似的K个用户，产生N个推荐
    def recommendByUser(self, user):
        K = self.user_n_sim_user
        N = self.user_n_rec_food
        rank = {}
        watched_food = self.trainSet[user]

        # v=similar user, wuv=similar factor
        for v, wuv in sorted(self.user_user_sim_matrix[user].items(), key=itemgetter(1), reverse=True)[0:K]:
            for food in self.trainSet[v]:
                if food in watched_food:
                    continue
                rank.setdefault(food, 0)
                rank[food] += wuv
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[0:N]

    def mergeRecommendResult(self,itemResult,userResult):

        # result = []
        # for item in itemResult:
        #     for i,user in enumerate(userResult):
        #         if item[0] == user[0]:
        #             result.append((item[0],user[1] + item[1]))
        #             break
        #         if i == len(userResult)-1:
        #             result.append((item[0], item[1]))
        #
        # for item in userResult:
        #     for i,user in enumerate(itemResult):
        #         if item[0] == user[0]:
        #             break
        #         if i == len(itemResult)-1:
        #             result.append((item[0], item[1]))
        # return result
        result = userResult[0:self.userWeight] + itemResult[0:self.itemWeight]
        return result


if __name__ == '__main__':
    rating_file = './datas/latest/ratings.csv'
    CF = CF()
    CF.get_dataset(rating_file)
    CF.item_calc_food_sim()

    CF.user_calc_user_sim()
    CF.evaluateMerge()
    CF.evaluateUser()
    CF.evaluateItem()
    # list1 = [('haha',1),('hehe',2),('yoyo',5)]
    # list2 = [('haha', 3),('yoyo',5),('gaga', 4)]
    # result = CF.mergeRecommendResult(list1,list2)
    # print(result)