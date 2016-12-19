#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "julius/juliuslib.h"
#include "plugin_defs.h"

#define PLUGIN_TITLE "A Feature Extractor for Okaeri Kanojo"

/* �v���g�^�C�v�R�[�� */
int startup(void *data);
int get_plugin_info(int opcode, char *buf, int buflen);
void func_end(Recog *recog, void *dummy);

/* �X�^�[�g�A�b�v�֐�. �R�[���o�b�N��o�^���� */
int startup(void *data) {
  Recog *recog = data;
  callback_add(recog, CALLBACK_EVENT_RECOGNITION_END, func_end, NULL);
  return 0;
}

/* �v���O�C�������`���� (�K���K�v) */
int get_plugin_info(int opcode, char *buf, int buflen) {
  switch(opcode) {
  case 0:
    /* plugin description string */
    strncpy(buf, PLUGIN_TITLE, buflen);
    break;
  }

  return 0;
}

/* end of the recognition */
void func_end(Recog *recog, void *dummy) {
  int i, j;
  int samplenum = recog->lmlist->am->mfcc->param->samplenum;

  /* get the mfcc & dmfcc values. */
  float **mfcc = recog->lmlist->am->mfcc->param->parvec;
  int mfcclen = recog->lmlist->am->mfcc->param->veclen;
  float *avg_mfcc;
  
  /* ���������蓖�� */
  avg_mfcc = (float *) malloc(sizeof(float) * mfcclen);
  for(i=0; i<mfcclen; i++) avg_mfcc[i] = 0.0;

  /* mfcc �̏u�Ԓl�𕽋ω����� */
  for(i=0; i<samplenum; i++) {
    for(j=0; j<mfcclen; j++) {
      avg_mfcc[j] += mfcc[i][j];
    }
  }
  for(i=0; i<mfcclen; i++) {
    avg_mfcc[i] /= samplenum;
    printf("%.2f ", avg_mfcc[i]);
  }
  printf("\n");
  free(avg_mfcc);
}

/* end of file */
