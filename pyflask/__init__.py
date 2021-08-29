import os, sys
import pymysql
import json
from flask import Flask, escape, request, Response, g, make_response, current_app
from flask.templating import render_template
from werkzeug.utils import secure_filename
from flask import Flask, escape, request, Response, g, make_response, current_app, Blueprint, session, redirect, url_for
#from neural_style_transfer import *
from . import g_prediction_tmp
import logging
import uuid
from PIL import Image
from io import BytesIO
import re, time, base64

bp = Blueprint('main', __name__, url_prefix='/')

@bp.before_app_request
def load_logged_in_user():
	user_id = session.get('username')
	if user_id is None:
		g.user = None
	else:
		g.user = user_id

app = Flask(__name__, static_url_path='/static')
app.debug = True

app.register_blueprint(bp)
app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'

# Main page
@app.route('/')
def index():
	return render_template('index.html')

@app.route('/login', methods=['GET','POST'])
def login():
	session.clear()
	if  request.method == 'POST':
		# User id
		u_id = request.form['user_id']
		
		# User pwd
		u_pwd = request.form['user_pwd']

		# MySQL
		db_user_id = sql_for_login(u_id, u_pwd)

		if not db_user_id:
			return render_template('login.html', no_user = "True")
		else:
			if 'username' in session:
				username = '%s' % escape(session['username'])
			else:
				session['username'] = u_id
				username = '%s' % escape(session['username'])
				return redirect(url_for('index'))

	return render_template('login.html')

@app.route('/logout')
def logout():
	session.clear()
	return redirect(url_for('login'))

@app.route('/img_select')
def img_select():
	return render_template('img_select.html')

@app.route('/result', methods=['GET','POST'])
def result():
	user_id = session.get('username')
	if  request.method == 'POST':
		# User Name
		g_user_name = request.form['pname']
		logging.error(g_user_name)
		# User Name
		g_user_birth = request.form['pbirth']
		
		# User Name
		g_user_tel = request.form['ptel']
		
		# User Image (target image)
		create_num = uuid.uuid4()

		g_user_img = request.files['user_img']
		#g_user_img.save('./pyflask/static/pyimages/'+str(g_user_img.filename))
		g_user_img.save('./pyflask/static/pyimages/'+str(create_num)+str(g_user_img.filename))
		g_user_img_path = './static/pyimages/'+str(create_num)+str(g_user_img.filename)
		g_user_img_path2 = './pyimages/'+str(create_num)+str(g_user_img.filename)

		# GoogLeNet Covid Prediction 
		g_pred_class, g_filepath, g_class_idx_list, binary_pred_class = g_prediction_tmp.main(g_user_img_path, str(create_num)+str(g_user_img.filename))

		if (binary_pred_class == 0):
			return render_template('img_select.html', binary_pred=binary_pred_class)
		else:
			g_class_idx_fst = str(g_class_idx_list[0])+'%'
			g_class_idx_snd = str(g_class_idx_list[1])+'%'
			
			print('init: '+g_filepath)
			
			if g_pred_class == 0:
				g_str_class = "정상"
			elif g_pred_class == 1:
				g_str_class = "코로나"
			elif g_pred_class == 2:
				g_str_class = "박테리아"
			elif g_pred_class == 3:
				g_str_class = "바이러스"
			print('g_str_class: '+g_str_class)

			# MySQL
			sql_for_result(g_user_name, g_user_tel, g_user_birth, g_user_img_path2, g_filepath, g_str_class, g_class_idx_fst, user_id)
	
	return render_template('result.html', p_name=g_user_name, p_birth=g_user_birth, p_tel=g_user_tel, refer_img=g_user_img_path2, user_img=g_user_img_path2, transfer_img=g_filepath, pred=g_str_class, prob=g_class_idx_fst, dr_lable_holder=g_str_class, dr_memo_holder="메모입력 후 수정", canvas_img_dir="None")
	
@app.route('/canvas_upload', methods=['GET','POST'])
def canvas_upload():
	
	# xray, patient 테이블 추출
	templete_xray, templete_patient = sql_for_templete_select()

	if  request.method == 'POST':
		canvas_img_base64 = request.form['canvas_img']

		# result.html에서 canvas 이미지 업로드(base64이미지 디코딩해서 서버 폴더에 저장)
		base64_data = re.sub('^data:image/.+;base64,', '', canvas_img_base64)
		byte_data = base64.b64decode(base64_data)
		image_data = BytesIO(byte_data)
		img = Image.open(image_data)
		canvas_name = templete_xray[3].split('/')[3]
		canvas_path = './pyflask/static/pyimages/canvas_img/'+canvas_name
		canvas_db_path = './pyimages/canvas_img/'+canvas_name
		img.save(canvas_path, "PNG")

		# dr_lable 수정 뒤 db에 저장
		sql_for_canvas(canvas_db_path)
	
	dr_lable, dr_memo = sql_for_dr_select(templete_xray[0], templete_patient[0])
	
	return render_template('result.html', p_name=templete_patient[1], p_birth=templete_patient[2], p_tel=templete_patient[3], refer_img=templete_xray[2], user_img=templete_xray[2], transfer_img=templete_xray[3], pred=templete_xray[4], prob=templete_xray[5], dr_lable_holder=dr_lable[0], dr_memo_holder=dr_memo[0], canvas_img_dir=canvas_db_path)

@app.route('/dr_lable_edit', methods=['GET','POST'])
def dr_lable_edit():
	
	templete_xray, templete_patient = sql_for_templete_select()
	
	g_dr_lable = templete_xray[4]
	if  request.method == 'POST':
		g_dr_lable = request.form['dr_lable']
		sql_for_dr_lable(g_dr_lable)
	
	dr_lable, dr_memo = sql_for_dr_select(templete_xray[0], templete_patient[0])
	
	return render_template('result.html', p_name=templete_patient[1], p_birth=templete_patient[2], p_tel=templete_patient[3], refer_img=templete_xray[2], user_img=templete_xray[2], transfer_img=templete_xray[3], pred=templete_xray[4], prob=templete_xray[5], dr_lable_holder=dr_lable[0], dr_memo_holder=dr_memo[0], canvas_img_dir=templete_xray[10])
						
@app.route('/dr_memo_edit', methods=['GET','POST'])
def dr_memo_edit():
	templete_xray, templete_patient = sql_for_templete_select()
	
	g_dr_memo = "메모입력 후 수정"
	if  request.method == 'POST':
		g_dr_memo = request.form['dr_memo']
		sql_for_dr_memo(g_dr_memo, templete_xray[0], templete_patient[0])
	
	dr_lable, dr_memo = sql_for_dr_select(templete_xray[0], templete_patient[0])

	dr_memo_text=""
	if(dr_memo[0]=="첫 판독자"):
		dr_memo_text="메모 입력 후 수정"
	else:
		dr_memo_text = dr_memo[0]
		
	return render_template('result.html', p_name=templete_patient[1], p_birth=templete_patient[2], p_tel=templete_patient[3], refer_img=templete_xray[2], user_img=templete_xray[2], transfer_img=templete_xray[3], pred=templete_xray[4], prob=templete_xray[5], dr_lable_holder=dr_lable[0], dr_memo_holder=dr_memo_text, canvas_img_dir=templete_xray[10])
	
@app.route('/patient_progress', methods=['GET','POST'])
def patient_progress(): 
	u_name = ""
	user_birth = ""
	user_tel = ""
	data_set = []
	ym_list = []
	
	if  request.method == 'POST':
		# User Name
		u_name = request.form['pname']
		
		# User Name
		user_birth = request.form['pbirth']
		
		# User Name
		user_tel = request.form['ptel']

		# MySQL
		db_data = sql_for_progress(u_name, user_tel, user_birth)

		# Assign Data from DB to Variables
		if not db_data:   # 예외처리
			ym_list.append(json.dumps(0))
		else:
			for i, data in enumerate(db_data) :
				x_id = json.loads(json.dumps(data[0], ensure_ascii=False))
				p_id = json.loads(json.dumps(data[1], ensure_ascii=False))
				img_path = json.loads(json.dumps(data[2], ensure_ascii=False))
				box_img_path = json.loads(json.dumps(data[3], ensure_ascii=False))
				pred_lable = json.loads(json.dumps(data[4], ensure_ascii=False))
				pred_prob = json.loads(json.dumps(data[5], ensure_ascii=False))
				dr_lable = json.loads(json.dumps(data[6], ensure_ascii=False))
				dr_note = json.loads(json.dumps(data[7], ensure_ascii=False))
				tmp = (data[8].strftime('%Y년 %m월 %d일'.encode('unicode-escape').decode())).encode().decode('unicode-escape')
				date = json.loads(json.dumps(tmp, ensure_ascii=False))
				canvas_img_path = json.loads(json.dumps(data[9], ensure_ascii=False))

				ym_tmp1 = (data[8].strftime('%Y년 %m월'.encode('unicode-escape').decode())).encode().decode('unicode-escape')
				ym_tmp2 = json.loads(json.dumps(ym_tmp1, ensure_ascii=False))
				if(i == 0 or i == len(db_data) - 1) :
					ym_list.append(ym_tmp2)

				if (canvas_img_path != "None"):
					box_img_path = canvas_img_path

				# 0:원본경로	1:박스이미지경로	2:예측클래스	3:예측정확도	4:수정클래스	5:의사메모	  6:날짜
				one_row = [x_id, p_id, img_path, box_img_path, pred_lable, pred_prob, dr_lable, dr_note, date]
				data_set.append(one_row)
				#sql_patient_memo_select(x_id,p_id)
		 
		print("period : ", ym_list)

	return render_template('patient_progress.html',
							p_name=u_name, p_birth=user_birth, p_tel=user_tel, data_set=data_set, period=ym_list)

@app.route('/patient_detail/<int:x_id>')
@app.route('/patient_detail/<int:x_id>/<int:p_id>')
def patient_detail(x_id, p_id): 
	
	patient_list, xray_list, memo_sql = sql_patient_info_select(x_id, p_id)
	canvas_img = xray_list[3]
	memo_list = []

	if (xray_list[10] != "None"):
		canvas_img = xray_list[10]

	for i, data in enumerate(memo_sql) :
		m_id = json.loads(json.dumps(data[0], ensure_ascii=False))
		x_id = json.loads(json.dumps(data[1], ensure_ascii=False))
		p_id = json.loads(json.dumps(data[2], ensure_ascii=False))
		user_id = json.loads(json.dumps(data[3], ensure_ascii=False))
		dr_memo = json.loads(json.dumps(data[4], ensure_ascii=False))
		tmp = (data[5].strftime('%Y-%m-%d %H:%M:%S'.encode('unicode-escape').decode())).encode().decode('unicode-escape')
		date = json.loads(json.dumps(tmp, ensure_ascii=False))
		# 0:원본경로	1:박스이미지경로	2:예측클래스	3:예측정확도	4:수정클래스	5:의사메모	  6:날짜
		one_row = [m_id, x_id, p_id, user_id, dr_memo, date]
		memo_list.append(one_row)
		#sql_patient_memo_select(x_id,p_id)

	return render_template('patient_detail.html', x_id = x_id, p_id = p_id, p_name=patient_list[1], p_birth=patient_list[3], p_tel=patient_list[2], transfer_img=canvas_img, pred=xray_list[4], prob=xray_list[5], dr_lable_holder=xray_list[6], memo_list=memo_list, xray_date = xray_list[8])
			
@app.route('/patient_detail_memo', methods=['GET','POST'])
def patient_detail_memo(): 
	user_id = session.get('username')
	if  request.method == 'POST':
		x_id = request.form['x_id']
		p_id = request.form['p_id']
		m_id = request.form.getlist('m_id')
		dr_memo = request.form['dr_memo']
		
		
		sql_detail_memo(m_id,x_id,p_id,user_id, dr_memo)
		
		patient_list, xray_list, memo_sql = sql_patient_info_select(x_id, p_id)

		canvas_img = xray_list[3]
		if (xray_list[10] != "None"):
			canvas_img = xray_list[10]
		
		memo_list = []
		for i, data in enumerate(memo_sql) :
			m_id = json.loads(json.dumps(data[0], ensure_ascii=False))
			x_id = json.loads(json.dumps(data[1], ensure_ascii=False))
			p_id = json.loads(json.dumps(data[2], ensure_ascii=False))
			user_id = json.loads(json.dumps(data[3], ensure_ascii=False))
			dr_memo = json.loads(json.dumps(data[4], ensure_ascii=False))
			tmp = (data[5].strftime('%Y-%m-%d %H:%M:%S'.encode('unicode-escape').decode())).encode().decode('unicode-escape')
			date = json.loads(json.dumps(tmp, ensure_ascii=False))
			# 0:원본경로	1:박스이미지경로	2:예측클래스	3:예측정확도	4:수정클래스	5:의사메모	  6:날짜
			one_row = [m_id, x_id, p_id, user_id, dr_memo, date]
			memo_list.append(one_row)
			
	return render_template('patient_detail.html', x_id = x_id, p_id = p_id, p_name=patient_list[1], p_birth=patient_list[3], p_tel=patient_list[2], transfer_img=canvas_img, pred=xray_list[4], prob=xray_list[5], dr_lable_holder=xray_list[6], memo_list=memo_list, xray_date = xray_list[8])

def sql_for_login(uid, upwd):
	db = pymysql.connect(host='', 
					port=, 
					user='', 
					passwd='', 
					db='', 
					charset='utf8'
					)
	cursor = db.cursor()

	# Select user id from MySQL.user table
	try :
		sql_select_user = """SELECT user_id FROM user WHERE user_id = %s AND user_pwd = %s;"""
		cursor.execute(sql_select_user, (uid, upwd))
		user_id = cursor.fetchall()
		db.commit()
		db.close()
		
		return user_id
	except TypeError as e :    # 에러처리(DB와 일치하는 데이터 없을 경우)
		return "None"

def sql_for_canvas(canvas_img_path):
	# DB Connect
	db = pymysql.connect(host='', 
					port=, 
					user='', 
					passwd='', 
					db='', 
					charset='utf8'
					)
	cursor = db.cursor()
	
	sql_update_dr_lable = """UPDATE xray SET x_canvas_img_path = %s WHERE x_id = (SELECT x_id FROM (SELECT MAX(x_id) AS x_id FROM xray) AS x_t);"""
	cursor.execute(sql_update_dr_lable, (canvas_img_path))
	db.commit()
	db.close()
	
def sql_for_result(name, tel, birth, i_path, box_path, lable, prob, user_id):
	# DB Connect
	db = pymysql.connect(host='', 
					port=, 
					user='', 
					passwd='', 
					db='', 
					charset='utf8'
					)
	cursor = db.cursor()
	
	# Insert Patient Info into MySQL.patient table
	sql_patient = """
					INSERT patient (p_name, p_phone, p_birth) 
					SELECT %s, %s, %s 
					WHERE NOT EXISTS (
						SELECT * 
						FROM patient 
						WHERE p_name = %s 
						AND p_phone = %s 
						AND p_birth = %s
					);
				"""
	cursor.execute(sql_patient, (name, tel, birth, name, tel, birth))
	db.commit()
	
	# Select p_id from MySQL.patient table
	sql_select_id = """SELECT p_id FROM patient WHERE p_phone = %s AND p_name = %s;"""
	cursor.execute(sql_select_id, (tel, name))
	pt_id = cursor.fetchall()
	db.commit()
			
	# Insert X-Ray Image Info into MySQL.xray table
	sql_xray = """
				INSERT INTO xray (p_id, x_img_path, x_box_img_path, x_pred_lable, x_pred_prob) 
				VALUES (%s, %s, %s, %s, %s);
				"""
	cursor.execute(sql_xray, (pt_id, i_path, box_path, lable, prob))
	
	sql_xray_select_id = """SELECT x_id FROM xray WHERE x_id = (SELECT x_id FROM (SELECT MAX(x_id) AS x_id FROM xray) AS x_t);"""
	cursor.execute(sql_xray_select_id)
	x_id = cursor.fetchall()
	
	sql_memo = """
				INSERT INTO memo (x_id, p_id, user_id, dr_memo) 
				VALUES (%s, %s, %s, %s);
				"""
	cursor.execute(sql_memo, (x_id, pt_id, user_id, "첫 판독자"))
				
	db.commit()
	db.close()
   
def sql_for_dr_lable(g_dr_lable):
	# DB Connect
	db = pymysql.connect(host='', 
					port=, 
					user='', 
					passwd='', 
					db='', 
					charset='utf8'
					)
	cursor = db.cursor()
	
	sql_update_dr_lable = """UPDATE xray SET x_dr_lable = %s WHERE x_id = (SELECT x_id FROM (SELECT MAX(x_id) AS x_id FROM xray) AS x_t);"""
	cursor.execute(sql_update_dr_lable, (g_dr_lable))
	db.commit()
	db.close()

def sql_for_dr_memo(g_dr_memo, x_id, p_id):
	# DB Connect
	db = pymysql.connect(host=' ', 
					port= , 
					user=' ', 
					passwd=' ', 
					db=' ', 
					charset='utf8'
					)
	cursor = db.cursor()
	
	sql_update_dr_memo = """UPDATE memo SET dr_memo = %s WHERE (x_id = %s AND p_id = %s);"""
	cursor.execute(sql_update_dr_memo, (g_dr_memo, x_id, p_id))
	db.commit()
	db.close()
   
def sql_for_dr_select(x_id, p_id):
	# DB Connect
	db = pymysql.connect(host=' ', 
					port= , 
					user=' ', 
					passwd=' ', 
					db=' ', 
					charset='utf8'
					)
	cursor = db.cursor()
	
	sql_dr_lable_select = """SELECT x_dr_lable, x_dr_note FROM xray WHERE x_id = (SELECT x_id FROM (SELECT MAX(x_id) AS x_id FROM xray) AS x_t);"""
	cursor.execute(sql_dr_lable_select)
	dr_lable_select = cursor.fetchone()
	
	sql_dr_memo_select = """SELECT dr_memo FROM memo WHERE (x_id = %s AND p_id = %s);"""
	cursor.execute(sql_dr_memo_select, (x_id, p_id))
	dr_memo_select = cursor.fetchone()
	
	db.commit()
	db.close()
	return dr_lable_select, dr_memo_select
   
def sql_for_templete_select():
	# DB Connect
	db = pymysql.connect(host=' ', 
					port= , 
					user=' ', 
					passwd=' ', 
					db=' ', 
					charset='utf8'
					)
	cursor = db.cursor()
	
	sql_xray_select = """SELECT * FROM xray WHERE x_id = (SELECT x_id FROM (SELECT MAX(x_id) AS x_id FROM xray) AS x_t);"""
	cursor.execute(sql_xray_select)
	xray_select = cursor.fetchone()
	p_id = xray_select[1]
	
	sql_patient_select = """SELECT * FROM patient WHERE p_id = %s;"""
	cursor.execute(sql_patient_select, (p_id))
	patient_select = cursor.fetchone()
	
	db.commit()
	db.close()
	return xray_select, patient_select

def sql_for_progress(name, tel, birth):
	db = pymysql.connect(host=' ', 
					port= , 
					user=' ', 
					passwd=' ', 
					db=' ', 
					charset='utf8'
					)
	cursor = db.cursor()

	# Select p_id from MySQL.patient table
	sql_select_id2 = """SELECT p_id FROM patient WHERE p_phone = %s AND p_name = %s AND p_birth = %s;"""
	cursor.execute(sql_select_id2, (tel, name, birth))
	pt_id2 = cursor.fetchall()
	db.commit()

	# Select Patient Img Data from MySQL.xray table
	try :
		sql_select_imgs = """SELECT x_id, p_id, x_img_path, x_box_img_path, x_pred_lable, x_pred_prob, x_dr_lable, x_dr_note, x_date, x_canvas_img_path FROM xray WHERE p_id = %s ORDER BY(-`x_id`);"""
		cursor.execute(sql_select_imgs, (pt_id2))
		pt_imgs = cursor.fetchall()
		db.commit()
		db.close()

		print(pt_imgs)
		print(len(pt_imgs))
		
		return pt_imgs
	except TypeError as e :    # 에러처리(DB와 일치하는 데이터 없을 경우)
		return pt_id2
		
def sql_patient_info_select(x_id, p_id):
	db = pymysql.connect(host=' ', 
					port= , 
					user=' ', 
					passwd=' ', 
					db=' ', 
					charset='utf8'
					)
	cursor = db.cursor()
	
	# Select p_id from MySQL.patient table
	sql_select_patient = """SELECT * FROM patient WHERE p_id = %s;"""
	cursor.execute(sql_select_patient, (p_id))
	patient_list = cursor.fetchone()
	db.commit()
	
	# Select p_id from MySQL.patient table
	sql_select_xray = """SELECT * FROM xray WHERE x_id = %s AND p_id = %s;"""
	cursor.execute(sql_select_xray, (x_id,p_id))
	xray_list = cursor.fetchone()
	db.commit()


	# Select p_id from MySQL.patient table
	sql_select_memo = """SELECT * FROM memo WHERE x_id = %s AND p_id = %s;"""
	cursor.execute(sql_select_memo, (x_id,p_id))
	memo_list = cursor.fetchall()
	db.commit()

	# Select Patient Img Data from MySQL.xray table
	return patient_list, xray_list, memo_list
		
def sql_detail_memo(m_id, x_id, p_id, user_id, dr_memo):
	db = pymysql.connect(host=' ', 
					port= , 
					user=' ', 
					passwd=' ', 
					db=' ', 
					charset='utf8'
					)
	cursor = db.cursor()
	
	if (m_id[0]=='0'):
		sql_memo_insert = """
				INSERT INTO memo (x_id, p_id, user_id, dr_memo) 
				VALUES (%s, %s, %s, %s);
				"""
		cursor.execute(sql_memo_insert, (x_id, p_id, user_id, dr_memo))
		db.commit()
	else:
		sql_memo_update = """UPDATE memo SET dr_memo = %s WHERE (m_id = %s);"""
		cursor.execute(sql_memo_update, (dr_memo, m_id))
		db.commit()
	# Select Patient Img Data from MySQL.xray table
	db.close()
	return 0