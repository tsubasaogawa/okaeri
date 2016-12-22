#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "julius/juliuslib.h"
#include "plugin_defs.h"

#define PLUGIN_TITLE "Okaeri Kanojo"
#define MAX_RESULT_LEN 2048
#define ALWAYS_OUT_MFCC_TO_FILE 1 /* �����ʃt�@�C������ɍ쐬���邩 */
#define CLASSIFIER_FILE "./classify.py"

/* �v���g�^�C�v�R�[�� */
int startup(void *data);
int get_plugin_info(int opcode, char *buf, int buflen);
void output_result(Recog *recog, void *dummy);
void output_mfccs_to_csv(char *filename, float *mfcc, int len);

/* �X�^�[�g�A�b�v�֐�. �R�[���o�b�N��o�^���� */
int startup(void *data) {
  Recog *recog = data;
  callback_add(recog, CALLBACK_RESULT, output_result, NULL);  
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
  char result[MAX_RESULT_LEN], mfcc_csv_name[256], classification_cmd[256];

  /* �F�����ʎ擾 */
  for(r=recog->process_list; r; r=r->next) {
    if(! r->live || r->result.status < 0) continue;

    /* ����̓x�X�g�̌��̂ݎ擾���� */
    winfo = r->lm->winfo;
    sent = &(r->result.sent[0]);
    wid = sent->word;
    for(i=0; i<sent->word_num; i++) { 
      printf("%s ", winfo->woutput[wid[i]]);
      sprintf(result, "%s%s", result, winfo->woutput[wid[i]]);
    }
    printf("\n");
  }

  /* �F�����ʔ��� */
  if(strstr(result, "��������") != NULL) {
    printf("tadaima ok\n");
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
  sprintf(mfcc_csv_name, "mfcc_%04d%02d%02d_%02d%02d%02d.csv", dt_data->tm_year + 1900, dt_data->tm_mon + 1, dt_data->tm_mday,
		  dt_data->tm_hour, dt_data->tm_min, dt_data->tm_sec);
  output_mfccs_to_csv(mfcc_csv_name, avg_mfcc, mfcclen);

  /* �b�ҔF������Ă�. �������艹���������ōĐ������ */
  sprintf(classification_cmd, CLASSIFIER_FILE "--testfile=%s", mfcc_csv_name);
  system(classification_cmd);

  free(avg_mfcc);
#if ALWAYS_OUT_MFCC_TO_FILE != 1
  remove(mfcc_csv_name);
#endif
}

void output_mfccs_to_csv(char *filename, float *mfcc, int len) {
  FILE *fp;
  int i;

  fp = fopen(filename, "w");
  if(fp == NULL) { exit(1); }

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