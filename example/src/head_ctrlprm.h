#ifndef _HEAD_CTRLPRM_
#define _HEAD_CTRLPRM_

typedef struct {
	double	dA[2];
	double	dB[2];
	double	dInPre;
	double	dOutPre;
} TF1_INF;						// 1st order TF information

typedef struct {
	double	dA[3];
	double	dB[3];
	double	dInPre[2];
	double	dOutPre[2];
} TF2_INF;						// 2nd order TF information

typedef struct {
	double	dA[4];
	double	dB[4];
	double	dInPre[3];
	double	dOutPre[3];
} TF3_INF;						// 3rd order TF information

extern TF2_INF	gstPIDInf[6];

extern TF1_INF	gstPDInf[6];

#endif
