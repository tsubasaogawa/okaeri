#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "julius/juliuslib.h"
#include "plugin_defs.h"

#define PLUGIN_TITLE "Okaeri Kanojo"
#define MAX_RESULT_LEN 2048
#define ALWAYS_OUT_MFCC_TO_FILE 0 /* �����ʃt�@�C������ɍ쐬���邩. 1 �ɂ���Ƙb�ҔF�����s���Ȃ� */
#define CLASSIFIER_FILE "python classify.py"
#define CLASSIFIER_PARAM "--unit=500"
#define OKAERI_CLASS 0

/* �v���g�^�C�v�R�[�� */
int startup(void *data);
int get_plugin_info(int opcode, char *buf, int buflen);
void output_result(Recog *recog, void *dummy);
void output_test_data_csv(char *filename, int cls, float *mfcc, int len);

/* �X�^�[�g�A�b�v�֐�. �R�[���o�b�N��o�^���� */
int startup(void *data) {
  Recog *recog = data;
  callback_add(recog, CALLBACK_RESULT, output_result, NULL);  
  setvbuf(stdout, 0, _IONBF, 0);
  return 0;
}

/* �v���O�C�������`���� (�K���K�v) */
int get_plugin_info(int opcode, char *buf, int buflen) {
  switch(opcode) {
  case 0:
    strncpy(buf, PLUGIN_TITLE, buflen);
    break;
  }

  return 0;
}

/* �F�����ʊm�莞�ɌĂ΂�� */
void output_result(Recog *recog, void *dummy) {
  int i, j;
  RecogProcess *r;
  WORD_INFO *winfo;
  WORD_ID *wid;
  Sentence *sent;
  int samplenum = recog->lmlist->am->mfcc->param->samplenum;
  time_t dt;
  struct tm *dt_data;
  float **mfcc = recog->lmlist->am->mfcc->param->parvec;
  int mfcclen = recog->lmlist->am->mfcc->param->veclen;
  float *avg_mfcc;
  char ph_prog_result[256], ph_result[MAX_RESULT_LEN], test_csv_name[256];
#if ALWAYS_OUT_MFCC_TO_FILE != 1
  char classification_cmd[256];
#endif
  
  /* �F�����ʎ擾 */
  ph_result[0] = '\0';
  for(r=recog->process_list; r; r=r->next) {
    if(! r->live || r->result.status < 0) continue;

    /* ����̓x�X�g�̌��̂ݎ擾���� */
    winfo = r->lm->winfo;
    sent = &(r->result.sent[0]);
    wid = sent->word;

    for(i=0; i<sent->word_num; i++) { 
      for(j=0; j<winfo->wlen[wid[i]]; j++) {
        center_name(winfo->wseq[wid[i]][j]->name, ph_prog_result);
        sprintf(ph_result, "%s%s", ph_result, ph_prog_result);
      }
    }
  }

  /* �F�����ʔ��� */
  if(strstr(ph_result, "tadaima") != NULL) {
    printf("okaeri_kanojo: tadaima recognized\n");
  }
#if ALWAYS_OUT_MFCC_TO_FILE != 1
  else {
    /* ��� mfcc ���o�͂��Ȃ��ݒ�ł���΂���ŏI�� */
    return;
  }
#endif

  /* mfcc �ۑ��p�z��̏����� */
  avg_mfcc = (float *) malloc(sizeof(float) * mfcclen);

  for(i=0; i<mfcclen; i++) {
    avg_mfcc[i] = 0.0;
  }

  /* mfcc �̏u�Ԓl�𔭘b���ω����� */
  for(i=0; i<samplenum; i++) {
    for(j=0; j<mfcclen; j++) {
      avg_mfcc[j] += mfcc[i][j];
    }
  }
  for(i=0; i<mfcclen; i++) {
    avg_mfcc[i] /= samplenum;
  }

  /* �v�Z���� mfcc ���t�@�C���ɏo�� */
  dt = time(NULL);
  dt_data = localtime(&dt);
  sprintf(test_csv_name, "test/test_%04d%02d%02d_%02d%02d%02d.csv", dt_data->tm_year + 1900, dt_data->tm_mon + 1, dt_data->tm_mday,
		  dt_data->tm_hour, dt_data->tm_min, dt_data->tm_sec);
  output_test_data_csv(test_csv_name, OKAERI_CLASS, avg_mfcc, mfcclen);

  /* �b�ҔF������Ă�. �������艹���������ōĐ������
   * �]�k: �{���̓v���O�C�����ŉ����Đ����������������]�͂��Ȃ� TODO */
#if ALWAYS_OUT_MFCC_TO_FILE != 1
  sprintf(classification_cmd, CLASSIFIER_FILE " " CLASSIFIER_PARAM " --testfile=%s", test_csv_name);
  system(classification_cmd);

  remove(test_csv_name);
#endif
  free(avg_mfcc);
}

/* �e�X�g�f�[�^�Ƃ��ĕۑ����� */
void output_test_data_csv(char *filename, int cls, float *mfcc, int len) {
  FILE *fp;
  int i;

  fp = fopen(filename, "w");
  if(fp == NULL) { exit(1); }

  /* �擪�ɃN���X��ǉ� */
  fprintf(fp, "%d,", cls);

  /* �ȍ~�� mfcc ��ǉ� */
  for(i=0; i<len; i++) {
    fprintf(fp, "%.3f", mfcc[i]);
    if(i != len - 1) {
      fprintf(fp, ",");
    }
  }
  fprintf(fp, "\n");
  fclose(fp);
}

/* end of file */
