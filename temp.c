
#include <stdio.h>
int main() {
    // CCarBilboard_2View.cpp : implementation of the CCarBilboard_2View class
//

#include "pch.h"
#include "framework.h"
// SHARED_HANDLERS can be defined in an ATL project implementing preview, thumbnail
// handlers and property sheets
#ifndef SHARED_HANDLERS
#include "CarBilboard_2.h"
#endif

#include "CarBilboard_2Doc.h"
#include "CarBilboard_2View.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CCarBilboard_2View

IMPLEMENT_DYNCREATE(CCarBilboard_2View, CView)

BEGIN_MESSAGE_MAP(CCarBilboard_2View, CView)
	// Standard printing commands
	ON_COMMAND(ID_FILE_PRINT, &CView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_DIRECT, &CView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_PREVIEW, &CView::OnFilePrintPreview)
	ON_WM_PAINT()
END_MESSAGE_MAP()

// CCarBilboard_2View construction/destruction

CCarBilboard_2View::CCarBilboard_2View() noexcept
{
	// TODO: add construction code here

}

CCarBilboard_2View::~CCarBilboard_2View()
{
}

BOOL CCarBilboard_2View::PreCreateWindow(CREATESTRUCT& cs)
{
	// TODO: Modify the Window class or styles here by modifying
	//  the CREATESTRUCT cs

	return CView::PreCreateWindow(cs);
}

// CCarBilboard_2View drawing

void CCarBilboard_2View::OnDraw(CDC* pDC)
{
	CCarBilboard_2Doc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	if (!pDoc)
		return;

	// TODO: add draw code for native data here
	pDC->TextOutW(10, 10, L"Welcome to the Car Billboard Application!");
	pDC->TextOutW(10, 30, L"Here you can display and print car billboards.");
}

// CCarBilboard_2View printing


void CCarBilboard_2View::OnFilePrintPreview()
{
#ifndef SHARED_HANDLERS
	AFXPrintPreview(this);
#endif
}

BOOL CCarBilboard_2View::OnPreparePrinting(CPrintInfo* pInfo)
{
	// default preparation
	return DoPreparePrinting(pInfo);
}

void CCarBilboard_2View::OnBeginPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: add extra initialization before printing
}

void CCarBilboard_2View::OnEndPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: add cleanup after printing
}

void CCarBilboard_2View::OnRButtonUp(UINT /*nFlags*/, CPoint point)
{
	ClientToScreen(&point);
	OnContextMenu(this, point);
}

void CCarBilboard_2View::DoPrinting(CDC* pDC, CPrintInfo* pInfo)
{
	// Set up the mapping mode and scaling
	pDC->SetMapMode(MM_ANISOTROPIC);
	pDC->SetWindowExt(1000, 1000);
	pDC->SetViewportExt(pDC->GetDeviceCaps(LOGPIXELSX), pDC->GetDeviceCaps(LOGPIXELSY));

	// Calculate the offset to center the output
	CRect rect;
	GetClientRect(&rect);
	int offsetX = (rect.Width() - pDC->GetDeviceCaps(LOGPIXELSX)) / 2;
	int offsetY = (rect.Height() - pDC->GetDeviceCaps(LOGPIXELSY)) / 2;
	pDC->SetViewportOrg(offsetX, offsetY);

	// Draw the content
	pDC->TextOutW(100, 100, L"Car Billboard");
	pDC->TextOutW(100, 200, L"Model: Example Model");
	pDC->TextOutW(100, 300, L"Year: 2023");
	pDC->TextOutW(100, 400, L"Price: $50,000");
}

void CCarBilboard_2View::OnPrint(CDC* pDC, CPrintInfo* pInfo)
{
	// Call the function to do the actual printing
	DoPrinting(pDC, pInfo);
}

// CCarBilboard_2View diagnostics

#ifdef _DEBUG
void CCarBilboard_2View::AssertValid() const
{
	CView::AssertValid();
}

void CCarBilboard_2View::Dump(CDumpContext& dc) const
{
	CView::Dump(dc);
}

CCarBilboard_2Doc* CCarBilboard_2View::GetDocument() const // non-debug version is inline
{
	ASSERT(m_pDocument->IsKindOf(RUNTIME_CLASS(CCarBilboard_2Doc)));
	return (CCarBilboard_2Doc*)m_pDocument;
}
#endif //_DEBUG


// CCarBilboard_2View message handlers
    return 0;
}
